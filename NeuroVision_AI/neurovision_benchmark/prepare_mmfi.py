"""Adapt MM-Fi official toolbox to the shared CSI CNN features.

THIS IS A SEPARATE TASK: sequence action classification, NOT binary presence.
MM-Fi is not pooled with WiMANS. Uses official MMFi sequence dataloader.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from nv_benchmark.features import csi_to_features


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--toolbox',type=Path,required=True,help='Clone of ybhbingo/MMFi_dataset')
    p.add_argument('--root',type=Path,required=True,help='MMFi actual downloaded data E01/...')
    p.add_argument('--config',type=Path,required=True,help='Official MMFi YAML with modality wifi-csi, data_unit sequence')
    p.add_argument('--output',type=Path,default=Path('data/processed/mmfi_actions.npz'))
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    sys.path.insert(0,str(args.toolbox.resolve()))
    from mmfi_lib.mmfi import make_dataset
    cfg=yaml.safe_load(args.config.read_text())
    if cfg.get('modality')!='wifi-csi' or cfg.get('data_unit')!='sequence':
        raise ValueError('MMFi config must use modality: wifi-csi and data_unit: sequence')
    train, val=make_dataset(str(args.root),cfg)
    samples=[]; observed=set()
    for dataset in (train,val):
        for i in range(len(dataset)):
            item=dataset[i]
            key=f"{item['scene']}/{item['subject']}/{item['action']}"
            if key in observed: continue
            observed.add(key)
            raw=np.asarray(item['input_wifi-csi'])
            # Official toolbox returns [T, Rx, subcarrier, window] in sequence mode.
            xs=csi_to_features(raw)
            samples.append((xs,str(item['action']),str(item['subject']),str(item['scene']),key))
            if args.limit and len(samples)>=args.limit: break
        if args.limit and len(samples)>=args.limit: break
    if not samples: raise RuntimeError('MM-Fi returned no samples')
    classes=sorted(set(s[1] for s in samples))
    ids={name:i for i,name in enumerate(classes)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    n=len(samples)
    np.savez_compressed(args.output,
        x=np.stack([r[0] for r in samples]),
        visual=np.zeros((n,8),np.float32),
        has_visual=np.zeros(n,dtype=bool),
        y=np.asarray([ids[r[1]] for r in samples],dtype=np.int64),
        ids=np.asarray([r[4] for r in samples]),
        groups=np.asarray([r[2] for r in samples]),
        environment=np.asarray([r[3] for r in samples]),
        class_names=np.asarray(classes),
        dataset=np.array('MM-Fi'),synthetic=np.array(False))
    print(f'Prepared {n} MM-Fi sequence samples with {len(classes)} action classes. '
          f'Group by subject, NOT by frame. No raw RGB in this export.')

if __name__=='__main__': main()
