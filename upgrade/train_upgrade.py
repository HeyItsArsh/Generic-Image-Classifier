"""Reproducible deduplicated transfer learning; test images never train the head."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KERAS_HOME']=str(__import__('pathlib').Path(__file__).resolve().parent/'keras_cache')
import json, hashlib, random, time, sys, tempfile, shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'upgrade'
(OUT/'temp').mkdir(exist_ok=True)
tempfile.tempdir=str(OUT/'temp')
CLASSES=json.loads((ROOT/'classes.json').read_text())
SEED=42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
EXT={'.jpg','.jpeg','.png','.bmp','.webp'}

def scan(source, aliases=None):
    print('Scanning',source,flush=True)
    rows=[]
    if not source.exists(): return rows
    for folder in sorted(source.iterdir()):
        name=(aliases or {}).get(folder.name,folder.name)
        if folder.is_dir() and name in CLASSES:
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix.lower() in EXT:
                    if len(rows)%500==0: print('scan',source.name,len(rows),flush=True)
                    rows.append((str(p),CLASSES.index(name),hashlib.sha256(p.read_bytes()).hexdigest()))
    return rows

def pixels(path,size):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)

def batches(rows,size):
    for start in range(0,len(rows),32):
        yield np.stack([pixels(r[0],size) for r in rows[start:start+32]])

def features(rows,base):
    arrays=[]
    for i,x in enumerate(batches(rows,224)):
        arrays.append(base(x/127.5-1,training=False).numpy())
        if i%20==0: print('features',i*32,'/',len(rows),flush=True)
    return np.concatenate(arrays)

def metrics(y,p):
    cm=np.zeros((len(CLASSES),len(CLASSES)),int)
    np.add.at(cm,(y,p.argmax(1)),1)
    rec=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(len(CLASSES)),where=cm.sum(1)>0)
    return {'accuracy':float(cm.trace()/cm.sum()),'balanced_accuracy':float(rec.mean()),'recall':dict(zip(CLASSES,rec.tolist())),'confusion_matrix':cm.tolist(),'images':int(cm.sum())}

def main():
    if '--reuse-split' in sys.argv:
        manifest=json.loads((OUT/'split_manifest.json').read_text())
        train,val,hold,original_test=([ (v['path'],CLASSES.index(v['class']),v['sha256']) for v in manifest[k]] for k in ('train','validation','holdout','existing_test'))
    else:
        original_test=scan(ROOT/'test_dataset')
        sources=scan(ROOT/'dataset')+scan(ROOT/'dataset_fruit')+scan(ROOT/'combined_dataset')+scan(ROOT/'test_dataset_Combined',{'cat':'cats','dog':'dogs'})
        test_hashes={r[2] for r in original_test}
        unique={}; conflicts=set(); duplicates=0
        for r in sources:
            if r[2] in unique:
                duplicates+=1
                if unique[r[2]][1]!=r[1]: conflicts.add(r[2])
            else: unique[r[2]]=r
        overlap=sum(h in test_hashes for h in unique)
        pool=[r for h,r in unique.items() if h not in test_hashes and h not in conflicts]
        train=[]; val=[]; hold=[]
        for label in range(len(CLASSES)):
            rows=[r for r in pool if r[1]==label]; random.shuffle(rows)
            # A separate unseen split from the broad animal collection complements the existing test set.
            nh=min(150,len(rows)//10); nv=min(100,len(rows)//10)
            hold+=rows[:nh]; val+=rows[nh:nh+nv]
            train+=rows[nh+nv:nh+nv+(1500 if CLASSES[label] in ('cats','dogs') else 900)]
        # Reject corrupt files once, before extracting or evaluating either model.
        bad=[]
        for rows in (train,val,hold,original_test):
            for r in list(rows):
                try: pixels(r[0],32)
                except Exception as e: bad.append([r[0],str(e)]); rows.remove(r)
        audit={'source_files':len(sources),'exact_duplicates':duplicates,'conflicting_hashes':len(conflicts),'source_test_overlap':overlap,'corrupt':bad,'counts':{k:dict(Counter(CLASSES[r[1]] for r in v)) for k,v in [('train',train),('validation',val),('holdout',hold),('existing_test',original_test)]}}
        (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
        (OUT/'split_manifest.json').write_text(json.dumps({k:[{'path':r[0],'class':CLASSES[r[1]],'sha256':r[2]} for r in v] for k,v in [('train',train),('validation',val),('holdout',hold),('existing_test',original_test)]},indent=2))
        print('AUDIT',json.dumps(audit),flush=True)
    base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')
    base.trainable=False
    data={}
    for name,rows in [('train',train),('validation',val),('holdout',hold),('existing_test',original_test)]:
        cache=OUT/(name+'_features.npz')
        signature=hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        meta=cache.with_suffix('.sha256')
        if cache.exists() and meta.exists() and meta.read_text()==signature:
            saved=np.load(cache); x=saved['x']; y=saved['y']
        else:
            x=features(rows,base); y=np.array([r[1] for r in rows]); np.savez_compressed(cache,x=x,y=y); meta.write_text(signature)
        data[name]=(x,y)
    x,y=data['train']; vx,vy=data['validation']
    # Feature standardization and a regularized linear head fit quickly on CPU.
    norm=tf.keras.layers.Normalization(); norm.adapt(x)
    head=tf.keras.Sequential([tf.keras.Input((1280,)),norm,tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(len(CLASSES),activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(.002))])
    head.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    counts=np.bincount(y,minlength=len(CLASSES)); weights={i:float(len(y)/(len(CLASSES)*n)) for i,n in enumerate(counts)}
    head.fit(x,y,validation_data=(vx,vy),batch_size=64,epochs=60,class_weight=weights,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(patience=7,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=3,factor=.5)])
    inputs=tf.keras.Input((224,224,3)); z=tf.keras.layers.Rescaling(1/127.5,offset=-1)(inputs); outputs=head(base(z,training=False)); model=tf.keras.Model(inputs,outputs,name='transfer_image_classifier')
    print('Exporting model',flush=True)
    export=Path(tempfile.gettempdir())/'agile_upgrade_export.keras'
    model.save(export); shutil.copy2(export,OUT/'image_classifier.keras'); export.unlink(); print('Export complete',flush=True); (OUT/'classes.json').write_text(json.dumps(CLASSES,indent=2))
    old=tf.keras.models.load_model(ROOT/'image_classifier.keras',compile=False)
    report={}
    for name,rows in [('validation',val),('holdout',hold),('existing_test',original_test)]:
        xx,yy=data[name]; newp=head.predict(xx,verbose=0)
        oldp=np.concatenate([old(b,training=False).numpy() for b in batches(rows,int(old.input_shape[1]))])
        report[name]={'new':metrics(yy,newp),'old':metrics(yy,oldp)}
        print('RESULT',name,json.dumps(report[name]),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__': main()
