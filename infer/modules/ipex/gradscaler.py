from collections import defaultdict
import torch
import intel_extension_for_pytorch as ipex                                               
import intel_extension_for_pytorch._C as core                                               

                                                                             
OptState = ipex.cpu.autocast._grad_scaler.OptState
_MultiDeviceReplicator = ipex.cpu.autocast._grad_scaler._MultiDeviceReplicator
_refresh_per_optimizer_state = (
    ipex.cpu.autocast._grad_scaler._refresh_per_optimizer_state
)


def _unscale_grads_(
    self, optimizer, inv_scale, found_inf, allow_fp16
):                                   
    per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
    per_device_found_inf = _MultiDeviceReplicator(found_inf)

                                                                                            
    per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))                               
                                
    if hasattr(optimizer, "sync_grad"):
        optimizer.sync_grad()
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if (not allow_fp16) and param.grad.dtype == torch.float16:
                    raise ValueError("Attempting to unscale FP16 gradients.")
                if param.grad.is_sparse:
                                                                                                      
                                                                                                   
                    if param.grad.dtype is torch.float16:
                        param.grad = param.grad.coalesce()
                    to_unscale = param.grad._values()
                else:
                    to_unscale = param.grad

                                                                                                     
                to_unscale = to_unscale.to("cpu")
                per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(
                    to_unscale
                )

        for _, per_dtype_grads in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                core._amp_foreach_non_finite_check_and_unscale_(
                    grads,
                    per_device_found_inf.get("cpu"),
                    per_device_inv_scale.get("cpu"),
                )

    return per_device_found_inf._per_device_tensors


def unscale_(self, optimizer):
    if not self._enabled:
        return

    self._check_scale_growth_tracker("unscale_")

    optimizer_state = self._per_optimizer_states[id(optimizer)]

    if optimizer_state["stage"] is OptState.UNSCALED:                                 
        raise RuntimeError(
            "unscale_() has already been called on this optimizer since the last update()."
        )
    elif optimizer_state["stage"] is OptState.STEPPED:
        raise RuntimeError("unscale_() is being called after step().")

                                                                                                         
    assert self._scale is not None
    inv_scale = (
        self._scale.to("cpu").double().reciprocal().float().to(self._scale.device)
    )
    found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
        optimizer, inv_scale, found_inf, False
    )
    optimizer_state["stage"] = OptState.UNSCALED


def update(self, new_scale=None):
    if not self._enabled:
        return

    _scale, _growth_tracker = self._check_scale_growth_tracker("update")

    if new_scale is not None:
                                          
        if isinstance(new_scale, float):
            self._scale.fill_(new_scale)                            
        else:
            reason = "new_scale should be a float or a 1-element torch.FloatTensor with requires_grad=False."
            assert isinstance(new_scale, torch.FloatTensor), reason                              
            assert new_scale.numel() == 1, reason
            assert new_scale.requires_grad is False, reason
            self._scale.copy_(new_scale)                            
    else:
                                                                                   
                                                                                                         
        found_infs = [
            found_inf.to(device="cpu", non_blocking=True)
            for state in self._per_optimizer_states.values()
            for found_inf in state["found_inf_per_device"].values()
        ]

        assert len(found_infs) > 0, "No inf checks were recorded prior to update."

        found_inf_combined = found_infs[0]
        if len(found_infs) > 1:
            for i in range(1, len(found_infs)):
                found_inf_combined += found_infs[i]

        to_device = _scale.device
        _scale = _scale.to("cpu")
        _growth_tracker = _growth_tracker.to("cpu")

        core._amp_update_scale_(
            _scale,
            _growth_tracker,
            found_inf_combined,
            self._growth_factor,
            self._backoff_factor,
            self._growth_interval,
        )

        _scale = _scale.to(to_device)
        _growth_tracker = _growth_tracker.to(to_device)
                                                                                            
    self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)


def gradscaler_init():
    torch.xpu.amp.GradScaler = ipex.cpu.autocast._grad_scaler.GradScaler
    torch.xpu.amp.GradScaler._unscale_grads_ = _unscale_grads_
    torch.xpu.amp.GradScaler.unscale_ = unscale_
    torch.xpu.amp.GradScaler.update = update
    return torch.xpu.amp.GradScaler

