# Kaggle sources verified — November 21, 2026

Kaggle datasets are **not included** in this repository. The official links may require a Kaggle account and acceptance of usage terms.

| Dataset | Direct page | Role | Caution |
| --- | --- | --- | --- |
| WiMANS | https://www.kaggle.com/datasets/shuokanghuang/wimans | PRIMARY paired CSI/video task | 2.4 and 5 GHz Wi-Fi measurements do not prove Wi-Fi 6 deployment performance; preserve alignment and session-level splits |
| CSI-Bench (updated release) | https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench | Additional CSI-only tasks: localization, proximity, activity, cross-device evaluation | Follow official code's versioned metadata/splits; no aligned RGB video documented in its baseline |
| WIFI CSI object detection | https://www.kaggle.com/datasets/armandokeller/wifi-csi-object-detection | Small exploratory object material/position classification | Small, apparatus-dependent and NOT dense 3D object detection |
| MM-Fi (official page, **not** Kaggle authoritative) | https://ntu-aiot-lab.github.io/mm-fi | CSI action/pose study | Official toolbox: https://github.com/ybhbingo/MMFi_dataset; raw RGB images subject to availability/privacy conditions |

## Kaggle CLI (optional)

Read https://github.com/Kaggle/kaggle-cli for the current installation/login procedure. Set up credentials through the Kaggle CLI; do not store API tokens in the project.

```bash
pip install kaggle
kaggle datasets download -d shuokanghuang/wimans -p data/downloads
kaggle datasets download -d guozhenjennzhu/csi-bench -p data/downloads
kaggle datasets download -d armandokeller/wifi-csi-object-detection -p data/downloads
```

Extract only under `data/raw/` (the `data/` directory is not bundled in a release). First inspect the archive entries, then verify the expected WiMANS root structure (`annotation.csv`, `wifi_csi/amp/`, `video/`). Do not guess that an archive is extracted into the right folder automatically. Dataset licenses and restrictions are controlled by their original publishers and must be checked before usage, especially for video and commercial applications.

WiMANS official: https://github.com/huangshk/WiMANS
CSI-Bench official code/release notes: https://github.com/johannayara/CSI-Bench
CSI object study: https://github.com/armandokeller/wifi-csi

## Why not merge all three?

These datasets differ in sensing setup, tasks, labels, CSI representation, and source hardware. Train **independent baselines** for WiMANS (paired video+CSI), MM-Fi (CSI action), CSI-Bench (CSI task) and object-detection data (object/material classification) before investigating transfer learning or domain adaptation. The provided source code currently implements WiMANS and MM-Fi adapters. CSI-Bench and object-study data are **research options, not implemented adapters**.
