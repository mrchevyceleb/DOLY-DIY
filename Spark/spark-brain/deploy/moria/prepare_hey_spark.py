from pathlib import Path
import io,json,urllib.request
import numpy as np
root=Path('/home/mrchevyceleb/spark-wake-training');assets=root/'assets';assets.mkdir(parents=True,exist_ok=True)
def download(url,path):
    if path.exists():return
    with urllib.request.urlopen(url,timeout=90) as response, path.with_suffix(path.suffix+'.part').open('wb') as out:
        while chunk:=response.read(1024*1024):out.write(chunk)
    path.with_suffix(path.suffix+'.part').replace(path)
    print('downloaded',path.name,path.stat().st_size,flush=True)
for name in ('embedding_model.onnx','melspectrogram.onnx'):
    download('https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'+name,assets/name)
for name in ('en_GB-vctk-medium.onnx','en_GB-vctk-medium.onnx.json'):
    download('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/vctk/medium/'+name,assets/name)
download('https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy',assets/'validation.npy')
if not (assets/'negative.npy').exists():
    req=urllib.request.Request('https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy',headers={'Range':'bytes=0-134217855'})
    with urllib.request.urlopen(req,timeout=90) as response:
        if response.status!=206:raise RuntimeError('Dataset server did not honor bounded Range request')
        data=response.read(134217856)
    stream=io.BytesIO(data);version=np.lib.format.read_magic(stream)
    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    shape,order,dtype=reader(stream)
    if order or shape[-1]!=96:raise RuntimeError('Unexpected negative embedding format')
    rows=(len(data)-stream.tell())//(96*dtype.itemsize)
    array=np.frombuffer(data,dtype=dtype,count=rows*96,offset=stream.tell()).reshape(rows,96)
    np.save(assets/'negative.npy',array)
    print('saved bounded negative feature subset',array.shape,flush=True)
print('READY',flush=True)
