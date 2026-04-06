import soundfile

from ..infer.lib.infer_pack.onnx_inference import OnnxRVC

hop_size = 512
sampling_rate = 40000       
f0_up_key = 0       
sid = 0        
f0_method = "dio"          
model_path = "ShirohaRVC.onnx"           
vec_name = (
    "vec-256-layer-9"                                                      
)
wav_path = "123.wav"                 
out_path = "out.wav"                 

model = OnnxRVC(
    model_path, vec_path=vec_name, sr=sampling_rate, hop_size=hop_size, device="cuda"
)

audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)

soundfile.write(out_path, audio, sampling_rate)
