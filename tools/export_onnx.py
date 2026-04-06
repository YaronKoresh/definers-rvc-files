import torch
from ..infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

if __name__ == "__main__":
    MoeVS = True                                 

    ModelPath = "Shiroha/shiroha.pth"        
    ExportedPath = "model.onnx"        
    hidden_channels = 256                              
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]         
    print(*cpt["config"])

    test_phone = torch.rand(1, 200, hidden_channels)               
    test_phone_lengths = torch.tensor([200]).long()                         
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)            
    test_pitchf = torch.rand(1, 200)         
    test_ds = torch.LongTensor([0])         
    test_rnd = torch.rand(1, 192, 200)              

    device = "cpu"                  

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False
    )                                           
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
                                                    
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
