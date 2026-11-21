"""YOLO inference only, no YOLO retraining or invented box annotations."""
import cv2
import numpy as np


def video_visual_features(video_path, yolo, sample_count: int = 3):
    """Eight measured features from sparse decoded frames of a WiMANS video.

    Order: mean people, max people, mean max confidence, left/center/right
    normalized person counts, mean person bbox area, observed video fraction.
    Not action recognition: snapshots alone cannot reliably detect walking.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Video could not be opened: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total-1, 0), sample_count).astype(int)
    records = []
    for ix in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ix))
        ok, frame = cap.read()
        if not ok:
            continue
        result = yolo.predict(frame, verbose=False, classes=[0], conf=.25)[0]
        xyxy = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0,4))
        confidences = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros(0)
        w, h = frame.shape[1], frame.shape[0]
        centers = (xyxy[:, 0] + xyxy[:, 2]) / (2 * w)
        areas = ((xyxy[:, 2]-xyxy[:, 0]) * (xyxy[:, 3]-xyxy[:, 1])) / (w*h)
        row = [len(centers), float(confidences.max()) if len(confidences) else 0.0,
               float((centers<1/3).sum()), float(((centers>=1/3)&(centers<2/3)).sum()),
               float((centers>=2/3).sum()), float(areas.mean()) if len(areas) else 0.0]
        records.append(row)
    cap.release()
    if not records:
        raise ValueError(f"No frames decoded from {video_path}")
    arr = np.asarray(records, dtype=np.float32)
    f = np.array([arr[:,0].mean(),arr[:,0].max(),arr[:,1].mean(),
                 arr[:,2].mean(),arr[:,3].mean(),arr[:,4].mean(),
                 arr[:,5].mean(),len(records)/sample_count],dtype=np.float32)
    return f
