"""Train a local wake head over openWakeWord/Google speech embeddings.

Run in a separate training environment; never opens a robot audio device.
Uses Piper voices and upstream negative features. Models are evaluated on
disjoint speaker IDs and a separate upstream ambient validation recording.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('OMP_NUM_THREADS', '3')
from pathlib import Path
import argparse
import json
import subprocess
import time
import wave
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt

ALIASES = ('Hey Spark.', 'Hey Sparks.', 'Hey Sparky.')
NEGATIVES = ('Spark.', 'Sparks.', 'Sparky.', 'Hey Mark.', 'Hey Park.',
             'Hey Stark.', 'A spark.', 'The park is open today.',
             'He sparks a discussion.', "What's the weather today?",
             'Tell me a joke.', 'Hey, how did you sleep?',
             'I like the bright lights.', 'Stop talking.')


def generate(root):
    assets, audio = root/'assets', root/'audio'
    audio.mkdir(exist_ok=True)
    manifest = []
    voices = [(assets/'en_GB-vctk-medium.onnx', range(109), 'vctk')]
    for name in ('hfc_female-medium', 'cori-high', 'lessac-high', 'kathleen-low'):
        prefix = 'en_GB' if name.startswith('cori') else 'en_US'
        voices.append((Path('/opt/piper-moria/voices')/f'{prefix}-{name}.onnx', [0], name))
    for model, speakers, voice in voices:
        entries = []
        for speaker in speakers:
            split = ('train' if speaker < 85 else 'val' if speaker < 97 else 'test') if voice == 'vctk' else (
                'test' if voice.startswith(('lessac', 'kathleen')) else 'train')
            texts = [(t, 1, 'plain', i) for i,t in enumerate(ALIASES)]
            texts += [(t+' Tell me a joke.', 1, 'command', i) for i,t in enumerate(ALIASES)]
            texts += [(t, 0, 'negative', i) for i,t in enumerate(NEGATIVES)]
            for text, label, kind, index in texts:
                path = audio/f'{voice}-{speaker}-{kind}-{index}.wav'
                entry = {'path': str(path), 'text': text, 'label': label,
                         'kind': kind, 'alias': index, 'voice': voice, 'speaker': speaker, 'split': split}
                manifest.append(entry)
                valid = False
                if path.exists():
                    try:
                        with wave.open(str(path)) as wav:
                            valid = wav.getnframes() > 0 and wav.getnchannels() == 1
                    except (EOFError, wave.Error):
                        pass
                if not valid:
                    request = {'text': text, 'output_file': str(path)}
                    if voice == 'vctk':request['speaker_id'] = speaker
                    entries.append(request)
        if entries:
            with (root/f'piper-{voice}.log').open('w') as log:
                subprocess.run(['/opt/piper-moria/piper/piper','-m',str(model),'--json-input'],
                    input='\n'.join(json.dumps(e) for e in entries)+'\n', text=True,
                    stdout=log, stderr=log, check=True)
        print('generated', voice, len(entries), flush=True)
    (root/'audio-manifest.json').write_text(json.dumps(manifest, indent=2))


def load_audio(path):
    with wave.open(str(path)) as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)
        rate = wav.getframerate()
    pcm = resample_poly(pcm, 16000, rate)
    # Drop Piper's leading and trailing quiet; keep the low-energy consonants.
    block = np.array([np.sqrt(np.mean(pcm[i:i+160]**2)) for i in range(0,len(pcm),160)])
    active = np.flatnonzero(block > max(20, block.max() * .015))
    if len(active):pcm=pcm[max(0,active[0]*160-320):min(len(pcm),(active[-1]+1)*160+320)]
    return pcm


def features(root):
    from openwakeword.utils import AudioFeatures
    assets = root/'assets'
    f = AudioFeatures(melspec_model_path=str(assets/'melspectrogram.onnx'),
        embedding_model_path=str(assets/'embedding_model.onnx'), inference_framework='onnx', ncpu=1)
    rng = np.random.default_rng(20261006)
    entries = json.loads((root/'audio-manifest.json').read_text())
    plain_lengths = {(e['voice'],e['speaker'],e['alias']):len(load_audio(e['path']))
                     for e in entries if e['kind']=='plain'}
    samples, labels, splits = [], [], []
    for ndx,e in enumerate(entries):
        pcm = load_audio(e['path'])
        if e['kind']=='command':
            pcm=pcm[:plain_lengths[e['voice'],e['speaker'],e['alias']]+1600]
        repeats = 6 if e['split']=='train' and e['label'] else 2
        for _ in range(repeats):
            speed = rng.uniform(.82,1.18)
            x=resample_poly(pcm,100,int(speed*100))
            x=x * (rng.uniform(70,3500)/max(1,np.sqrt(np.mean(x*x))))
            # Mild room reflections plus residual noise and robot high-pass.
            delay=int(rng.uniform(.03,.13)*16000)
            reflected=np.pad(x,(delay,0))[:len(x)]*rng.uniform(0,.25)
            x=x+reflected
            x=sosfilt(butter(2,150,fs=16000,btype='highpass',output='sos'),x)
            x=x+rng.normal(0,rng.uniform(3,35),len(x))
            x=np.clip(x,-32768,32767).astype(np.int16)
            before=int(rng.uniform(.2,.65)*16000)
            after=int(rng.uniform(.06,.3)*16000)
            clip=np.pad(x,(before,after))
            clip=clip[-48000:]
            clip=np.pad(clip,(max(0,48000-len(clip)),0))
            emb=f._get_embeddings(clip)
            samples.append(emb[-16:]);labels.append(e['label']);splits.append(e['split'])
        if ndx%200==0:print('features',ndx,'/',len(entries),flush=True)
    np.savez(root/'speech-features.npz',x=np.asarray(samples,dtype=np.float32),
             y=np.asarray(labels),split=np.asarray(splits))


def train(root):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    d=np.load(root/'speech-features.npz'); rng=np.random.default_rng(20261006)
    mask=d['split']=='train';x=d['x'][mask];y=d['y'][mask]
    # End-of-clip negatives miss prefixes such as "He sparks ...". Train on
    # their whole sliding timeline so the live detector rejects them early.
    hard_path = root/'streaming-negatives.npy'
    if not hard_path.exists():
        from openwakeword.utils import AudioFeatures
        f = AudioFeatures(melspec_model_path=str(root/'assets/melspectrogram.onnx'),
            embedding_model_path=str(root/'assets/embedding_model.onnx'),
            inference_framework='onnx', ncpu=1)
        negatives = []
        entries = json.loads((root/'audio-manifest.json').read_text())
        for e in entries:
            if e['split'] != 'train' or e['label']:
                continue
            pcm = load_audio(e['path'])
            for rms in (150, 1200):
                z=pcm*(rms/max(1,np.sqrt(np.mean(pcm*pcm))))
                z=sosfilt(butter(2,150,fs=16000,btype='highpass',output='sos'),z)
                z=np.pad(z,(24000,8000))
                z=np.clip(z+rng.normal(0,8,len(z)),-32768,32767).astype(np.int16)
                emb=f._get_embeddings(z)
                negatives.extend(emb[i:i+16] for i in range(0,len(emb)-15,2))
        np.save(hard_path,np.asarray(negatives,dtype=np.float32))
    hard=np.load(hard_path)
    x=np.concatenate((x,hard));y=np.concatenate((y,np.zeros(len(hard),dtype=int)))
    context_negative_path = root/'context-negatives.npy'
    if not context_negative_path.exists():
        from openwakeword.utils import AudioFeatures
        f=AudioFeatures(melspec_model_path=str(root/'assets/melspectrogram.onnx'),
            embedding_model_path=str(root/'assets/embedding_model.onnx'),inference_framework='onnx',ncpu=1)
        entries=json.loads((root/'audio-manifest.json').read_text())
        backgrounds=[e['path'] for e in entries if e['split']=='train'
                     and e['kind']=='negative' and e['alias']>=7]
        negatives=[]
        for e in entries:
            if e['split']!='train' or e['label']:continue
            pcm=load_audio(e['path'])
            rms=300
            z=pcm*(rms/max(1,np.sqrt(np.mean(pcm*pcm))))
            z=np.pad(z,(24000,16000))
            bg=load_audio(backgrounds[int(rng.integers(len(backgrounds)))])
            bg=np.tile(bg,1+len(z)//len(bg))[:len(z)]
            bg=bg*(rms*rng.uniform(.03,.55)/max(1,np.sqrt(np.mean(bg*bg))))
            z=sosfilt(butter(2,150,fs=16000,btype='highpass',output='sos'),z+bg)
            z=np.clip(z+rng.normal(0,8,len(z)),-32768,32767).astype(np.int16)
            emb=f._get_embeddings(z)
            negatives.extend(emb[i:i+16] for i in range(0,len(emb)-15,2))
        np.save(context_negative_path,np.asarray(negatives,dtype=np.float32))
    context_negative=np.load(context_negative_path)
    x=np.concatenate((x,context_negative));y=np.concatenate((y,np.zeros(len(context_negative),dtype=int)))
    positive_path = root/'context-positives.npy'
    if not positive_path.exists():
        from openwakeword.utils import AudioFeatures
        f = AudioFeatures(melspec_model_path=str(root/'assets/melspectrogram.onnx'),
            embedding_model_path=str(root/'assets/embedding_model.onnx'),
            inference_framework='onnx', ncpu=1)
        positives=[]
        entries=json.loads((root/'audio-manifest.json').read_text())
        backgrounds=[e['path'] for e in entries if e['split']=='train'
                     and e['kind']=='negative' and e['alias']>=7]
        for e in entries:
            if e['split'] != 'train' or e['kind'] != 'plain':
                continue
            pcm=load_audio(e['path'])
            for rms in (150,300,1200):
                z=pcm*(rms/max(1,np.sqrt(np.mean(pcm*pcm))))
                z=sosfilt(butter(2,150,fs=16000,btype='highpass',output='sos'),z)
                z=np.pad(z,(24000,16000))
                # Real standby includes room talk before and during a greeting.
                # Silence-only padding made the original head depend on a
                # quiet context, even when it recognized the isolated voice.
                bg=load_audio(backgrounds[int(rng.integers(len(backgrounds)))])
                bg=np.tile(bg,1+len(z)//len(bg))[:len(z)]
                bg=bg*(rms*rng.uniform(.03,.55)/max(1,np.sqrt(np.mean(bg*bg))))
                bg=sosfilt(butter(2,150,fs=16000,btype='highpass',output='sos'),bg)
                z=z+bg
                z=np.clip(z+rng.normal(0,8,len(z)),-32768,32767).astype(np.int16)
                emb=f._get_embeddings(z)
                greeting_end=1.5+len(pcm)/16000
                for i in range(len(emb)-15):
                    end=.76+(i+15)*.08
                    if greeting_end <= end <= greeting_end+.4:
                        positives.append(emb[i:i+16])
        np.save(positive_path,np.asarray(positives,dtype=np.float32))
    positive=np.load(positive_path)
    x=np.concatenate((x,positive));y=np.concatenate((y,np.ones(len(positive),dtype=int)))
    background=np.load(root/'assets/negative.npy',mmap_mode='r')
    starts=rng.integers(0,len(background)-16,24000)
    generic=np.array([background[s:s+16] for s in starts],dtype=np.float32)
    # Keep disjoint ambient regions: first third trains negative contexts,
    # second calibrates, final third evaluates without threshold selection.
    ambient=np.load(root/'assets/validation.npy',mmap_mode='r')
    # Cover the training region's continuous phases rather than a sparse
    # random sample that can miss short, confidently misclassified sounds.
    starts=range(0,len(ambient)//3-16,2)
    ambient_train=np.array([ambient[s:s+16] for s in starts],dtype=np.float32)
    generic=np.concatenate((generic,ambient_train))
    # Silence/noise embeddings also prevent an idle classifier from firing.
    x=np.concatenate((x,generic)); y=np.concatenate((y,np.zeros(len(generic),dtype=int)))
    pipe=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(96,),
        max_iter=90,batch_size=256,early_stopping=True,n_iter_no_change=8,
        learning_rate_init=.001,alpha=.02,random_state=20261006))
    with threadpool_limits(limits=3):
        pipe.fit(x.reshape(len(x),-1),y)
    scaler,mlp=pipe.steps[0][1],pipe.steps[1][1]
    weights=[numpy_helper.from_array(scaler.mean_.astype('float32'),'mean'),
             numpy_helper.from_array(scaler.scale_.astype('float32'),'scale')]
    nodes=[helper.make_node('Flatten',['features'],['flat'],axis=1),
           helper.make_node('Sub',['flat','mean'],['center']),
           helper.make_node('Div',['center','scale'],['normalized'])]
    last='normalized'
    for i,(w,b) in enumerate(zip(mlp.coefs_,mlp.intercepts_)):
        weights.extend((numpy_helper.from_array(w.astype('float32'),f'w{i}'),
                        numpy_helper.from_array(b.astype('float32'),f'b{i}')))
        nodes.extend((helper.make_node('MatMul',[last,f'w{i}'],[f'm{i}']),
                      helper.make_node('Add',[f'm{i}',f'b{i}'],[f'a{i}']),
                      helper.make_node('Relu' if i<len(mlp.coefs_)-1 else 'Sigmoid',
                                       [f'a{i}'],['score' if i==len(mlp.coefs_)-1 else f'h{i}'])))
        last=f'h{i}'
    graph=helper.make_graph(nodes,'hey_spark',[helper.make_tensor_value_info('features',TensorProto.FLOAT,[None,16,96])],
                            [helper.make_tensor_value_info('score',TensorProto.FLOAT,[None,1])],weights)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=8
    onnx.checker.check_model(model);onnx.save(model,root/'assets/hey_spark.onnx')
    report={}
    for split in ('val','test'):
        m=d['split']==split; scores=pipe.predict_proba(d['x'][m].reshape(sum(m),-1))[:,1];truth=d['y'][m]
        report[split]={'positive':int(sum(truth)),'negative':int(sum(truth==0)),
                       'min_positive':float(scores[truth==1].min()),'max_negative':float(scores[truth==0].max()),
                       'recall_at_08':float(np.mean(scores[truth==1]>=.8)),
                       'false_at_08':int(sum(scores[truth==0]>=.8))}
    report['head_iterations']=mlp.n_iter_;report['train_examples']=len(x)
    (root/'head-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','features','train','all'])
    p.add_argument('--root',type=Path,default=Path('/home/mrchevyceleb/spark-wake-training'))
    a=p.parse_args()
    for name,func in (('generate',generate),('features',features),('train',train)):
        if a.stage in ('all',name):
            started=time.monotonic();func(a.root);print(name,'seconds',round(time.monotonic()-started,1),flush=True)
