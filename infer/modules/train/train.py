import os
import sys
import logging
import datetime
from random import randint, shuffle
from time import sleep, time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

now_dir = os.getcwd()
sys.path.append(now_dir)

from ...lib.train import utils
from ...lib.infer_pack import commons
from ...lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from ...lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from ...lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from ...lib.train.process_ckpt import savee

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from ..ipex import ipex_init
        from ..ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast
        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

if hps.version == "v1":
    from ...lib.infer_pack.models import MultiPeriodDiscriminator
    from ...lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from ...lib.infer_pack.models import SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0
else:
    from ...lib.infer_pack.models import SynthesizerTrnMs768NSFsid as RVC_Model_f0
    from ...lib.infer_pack.models import SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0
    from ...lib.infer_pack.models import MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator

global_step = 0

class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | ({datetime.timedelta(seconds=elapsed_time)})"

def main():
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        n_gpus = 1
    if n_gpus < 1:
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    logger = utils.get_logger(hps.model_dir)
    mp.spawn(run, args=(n_gpus, hps, logger))

def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://?use_libuv=False", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        pass
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu", weights_only=True)["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu", weights_only=True)["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu", weights_only=True)["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu", weights_only=True)["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache):
    net_g, net_d = nets; optim_g, optim_d = optims
    train_loader, _ = loaders  # Eval loader not used in this shortened version
    if writers: writer, _ = writers  # Writer for TensorBoard

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step; net_g.train(); net_d.train()

    data_iterator = cache if hps.if_cache_data_in_gpu else enumerate(train_loader)
    if hps.if_cache_data_in_gpu and not cache:  # Populate cache if needed
        for batch_idx, info in train_loader:
            data = [i.cuda(rank, non_blocking=True) if torch.cuda.is_available() else i for i in info] # Move to device
            cache.append((batch_idx, data))
    elif hps.if_cache_data_in_gpu:
        shuffle(cache) # shuffle cache every epoch

    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        if hps.if_cache_data_in_gpu:
            batch_idx, info = info

        phone, phone_lengths, *rest, spec, spec_lengths, wave, _ = info  # Unpack data
        if hps.if_f0: pitch, pitchf, sid = rest # unpack f0 related data if needed
        else: sid = rest[0]

        with autocast(enabled=hps.train.fp16_run):
            gen_inputs = [phone, phone_lengths, spec, spec_lengths, sid]
            if hps.if_f0: gen_inputs = [phone, phone_lengths, pitch, pitchf, *gen_inputs]
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(*gen_inputs)

            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)

            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                if hps.train.fp16_run: y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False): loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad(); scaler.scale(loss_disc).backward(); scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None); scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad(); scaler.scale(loss_gen_all).backward(); scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None); scaler.step(optim_g); scaler.update()

        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]")
            logger.info([global_step, lr])
            logger.info(f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}")
            scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g,
                           "loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl}
            scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
            scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
            image_dict = {"slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                          "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                          "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())}
            utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)

        global_step += 1

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(5)
        os._exit(0)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()





