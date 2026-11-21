"""Prepare paired samples for binary occupancy presence (>=1 human), not actions.

WiMANS official annotation.csv columns 'label', 'number_of_users'.
CSI .npy array is time-first. Use real video YOLO features only with --with-yolo.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from nv_benchmark.features import csi_to_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='WiMANS dataset root')
    p.add_argument('--output', type=Path, default=Path('data/processed/wimans_presence.npz'))
    p.add_argument('--limit', type=int, default=0, help='0 = all samples')
    p.add_argument('--with-yolo', action='store_true')
    p.add_argument('--weights', default='yolo26n.pt')
    args = p.parse_args()
    annotations = args.root / 'annotation.csv'
    amp_dir = args.root / 'wifi_csi' / 'amp'
    video_dir = args.root / 'video'
    if not annotations.exists() or not amp_dir.is_dir():
        raise FileNotFoundError(f'Missing {annotations} or {amp_dir}')
    df = pd.read_csv(annotations, dtype=str)
    necessary = {'label', 'number_of_users'}
    if not necessary.issubset(df.columns):
        raise ValueError(f'Expected {necessary}, found {df.columns.tolist()}')
    if args.with_yolo:
        from ultralytics import YOLO
        from nv_benchmark.vision import video_visual_features
        detector = YOLO(args.weights)
    else:
        detector = None
    rows = []
    missing = 0
    for _, r in df.iterrows():
        sample_id = str(r['label'])
        path = amp_dir / f'{sample_id}.npy'
        if not path.is_file():
            missing += 1
            continue
        try:
            x = csi_to_features(np.load(path, allow_pickle=False))
            y = int(int(r['number_of_users']) > 0)
            camera = np.zeros(8, dtype=np.float32)
            has_camera = False
            if detector is not None:
                vid = video_dir / f'{sample_id}.mp4'
                if vid.is_file():
                    camera = video_visual_features(vid, detector)
                    has_camera = True
            # act_30_25 -> act_30; verify recording grouping for your protocol.
            group = sample_id.rsplit('_', 1)[0]
            rows.append((x, camera, y, sample_id, group,has_camera,
                         str(r.get('environment','unknown')),
                         str(r.get('wifi_band','unknown'))))
        except Exception as exc:
            raise RuntimeError(f'Failed sample {sample_id}: {exc}') from exc
        if args.limit and len(rows)>=args.limit:
            break
    if not rows:
        raise RuntimeError('No valid samples. Check WiMANS file paths and dataset access.')
    xs,cam,ys,ids,groups,present,env,bands = zip(*rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
        x=np.asarray(xs,dtype=np.float32),
        visual=np.asarray(cam,dtype=np.float32),
        y=np.asarray(ys,dtype=np.int64),
        ids=np.asarray(ids),groups=np.asarray(groups),
        has_visual=np.asarray(present,dtype=bool),
        environment=np.asarray(env), wifi_band=np.asarray(bands),
        dataset=np.array('WiMANS'), synthetic=np.array(False))
    print(json.dumps({'path':str(args.output),'n_samples':len(rows),
        'missing_csi_skipped':missing,'with_video':int(sum(present)),
        'positive':int(sum(ys))},indent=2))


if __name__=='__main__':
    main()
