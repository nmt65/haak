"""Synthetic fixture ONLY for integration tests, NOT a dataset benchmark."""
from pathlib import Path
import numpy as np
from nv_benchmark.features import csi_to_features


def main():
    root = Path('data/processed')
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x, v, y, ids, groups = [], [], [], [], []
    for i in range(96):
        label = (i % 8) // 4  # each independent group contains both synthetic classes
        t = np.linspace(0, 3, 220)
        signal = (0.3 * rng.standard_normal((220, 3, 20)) + 2.0).astype(np.float32)
        if label:
            signal += (0.7*np.sin(t*12))[:,None,None]
        x.append(csi_to_features(signal))
        v.append(np.array([label+rng.uniform(0,.2),label+rng.uniform(0,.2),
                           label*.8, label*.3,label*.3,label*.3,label*.1,1.0],np.float32))
        y.append(label)
        ids.append(f'synthetic_{i:03d}')
        groups.append(f'group_{i//8:02d}')
    np.savez_compressed(root/'synthetic_fixture.npz', x=np.asarray(x),
        visual=np.asarray(v), y=np.asarray(y,dtype=np.int64),
        ids=np.asarray(ids),groups=np.asarray(groups),
        has_visual=np.ones(len(y),dtype=bool),
        dataset=np.array('SYNTHETIC'),synthetic=np.array(True))
    print('Generated synthetic fixture: 96 samples, 12 groups, never real CSI')


if __name__=='__main__':
    main()
