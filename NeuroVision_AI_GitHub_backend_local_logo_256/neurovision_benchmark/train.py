"""Reproducible grouped benchmark, with train-only normalization and held-out test.

Tasks supported: binary presence (WiMANS) and multiclass activity (MM-Fi).
Model modes: csi / camera (YOLO features) / fusion / all (fair same cohort).
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from nv_benchmark.features import standardize_fit
from nv_benchmark.models import PerceptionClassifier


def grouped_split(groups, seed=42):
    ids = np.arange(len(groups))
    train_idx, remain_idx = next(GroupShuffleSplit(n_splits=1,test_size=.30,
                                                     random_state=seed).split(ids,groups=groups))
    val_rel, test_rel = next(GroupShuffleSplit(n_splits=1,test_size=.50,
                                random_state=seed+1).split(remain_idx,groups=np.asarray(groups)[remain_idx]))
    val_idx, test_idx = remain_idx[val_rel], remain_idx[test_rel]
    assert not (set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[val_idx]))
    assert not (set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx]))
    assert not (set(np.asarray(groups)[val_idx]) & set(np.asarray(groups)[test_idx]))
    return train_idx, val_idx, test_idx


def evaluate(model, csi, vis, y, indices, device, task, threshold=.5):
    model.eval()
    with torch.no_grad():
        logits = []
        for start in range(0,len(indices),128):
            batch = indices[start:start+128]
            val = model(torch.from_numpy(csi[batch]).to(device),
                        torch.from_numpy(vis[batch]).to(device))
            logits.append(val.detach().cpu().numpy())
    pred_logits = np.concatenate(logits,axis=0)
    labels = y[indices]
    if task == 'binary':
        probs = 1/(1+np.exp(-np.clip(pred_logits,-30,30)))
        preds = (probs>=threshold).astype(int)
        matrix = confusion_matrix(labels,preds,labels=[0,1])
        true_negatives, false_positives = int(matrix[0,0]), int(matrix[0,1])
        # Share of "no human" samples wrongly reported as "human present".
        negatives = true_negatives + false_positives
        false_positive_rate = false_positives / negatives if negatives else 0.0
        result = dict(accuracy=float(accuracy_score(labels,preds)),
            balanced_accuracy=float(balanced_accuracy_score(labels,preds)),
            macro_f1=float(f1_score(labels,preds,average='macro',zero_division=0)),
            f1=float(f1_score(labels,preds,zero_division=0)),
            precision=float(precision_score(labels,preds,zero_division=0)),
            recall=float(recall_score(labels,preds,zero_division=0)),
            brier=float(brier_score_loss(labels,probs)),
            false_positive_rate=float(false_positive_rate),
            confusion_matrix=matrix.tolist())
        return result, probs
    else:
        preds = pred_logits.argmax(axis=1)
        labels_all = list(range(pred_logits.shape[1]))
        result = dict(accuracy=float(accuracy_score(labels,preds)),
            balanced_accuracy=float(balanced_accuracy_score(labels,preds)),
            macro_f1=float(f1_score(labels,preds,average='macro',zero_division=0,
                                      labels=labels_all)),
            weighted_f1=float(f1_score(labels,preds,average='weighted',zero_division=0)),
            confusion_matrix=confusion_matrix(labels,preds,labels=labels_all).tolist())
        return result, pred_logits


def timed_inference(model, csi, vis, indices, device):
    model.eval()
    times = []
    with torch.no_grad():
        # single-sample latency after warming up
        idx = int(indices[0])
        a = torch.from_numpy(csi[idx:idx+1]).to(device)
        b = torch.from_numpy(vis[idx:idx+1]).to(device)
        for _ in range(4):
            _ = model(a,b)
        if device.type=='cuda': torch.cuda.synchronize()
        for idx in indices[:min(len(indices),40)]:
            a = torch.from_numpy(csi[idx:idx+1]).to(device)
            b = torch.from_numpy(vis[idx:idx+1]).to(device)
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            _=model(a,b)
            if device.type=='cuda': torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    return {'p50_ms':round(float(np.median(times)),3),
            'p95_ms':round(float(np.percentile(times,95)),3),
            'num_iterations':len(times),
            'note':'model forward only, not CSI capture, video decode or YOLO'}


def run_mode(mode, data, seed, epochs, batch_size, lr, device, run_dir, split, patience=3):
    x = data['x'].astype(np.float32)
    visual = data['visual'].astype(np.float32)
    y = data['y'].astype(np.int64)
    train_i, val_i, test_i = split
    mu, sd = standardize_fit(x[train_i])
    x = np.clip((x-mu)/sd,-10,10).astype(np.float32)
    v_mu=visual[train_i].mean(axis=0,keepdims=True)
    v_sd=np.maximum(visual[train_i].std(axis=0,keepdims=True),1e-5)
    visual=np.clip((visual-v_mu)/v_sd,-10,10).astype(np.float32)
    n_classes=int(np.max(y)+1)
    if not np.array_equal(np.unique(y),np.arange(n_classes)):
        raise ValueError('Class labels must be contiguous integers starting at 0')
    task='binary' if n_classes==2 else 'multiclass'
    model=PerceptionClassifier(mode,channels=x.shape[1],visual_dim=visual.shape[1],
                               n_outputs=1 if task=='binary' else n_classes).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    loss_fn=nn.BCEWithLogitsLoss() if task=='binary' else nn.CrossEntropyLoss()
    ds=TensorDataset(torch.from_numpy(x[train_i]),torch.from_numpy(visual[train_i]),
                     torch.from_numpy(y[train_i]))
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True)
    best=-1; best_state=None; history=[]
    epochs_without_improvement=0
    stopped_early_at=None
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for c,v,labels in loader:
            c,v,labels=c.to(device),v.to(device),labels.to(device)
            optimizer.zero_grad()
            logits=model(c,v)
            loss=loss_fn(logits,labels.float() if task=='binary' else labels.long())
            loss.backward(); optimizer.step(); losses.append(loss.item())
        val_metrics,_=evaluate(model,x,visual,y,val_i,device,task)
        history.append({'epoch':epoch,'loss':round(float(np.mean(losses)),5),
                        'val_macro_f1':val_metrics['macro_f1']})
        if val_metrics['macro_f1']>best:
            best=val_metrics['macro_f1']
            best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
            epochs_without_improvement=0
        else:
            epochs_without_improvement+=1
        # Early stopping: validation macro-F1 has not improved for `patience` epochs.
        if patience and epochs_without_improvement>=patience:
            stopped_early_at=epoch
            break
    model.load_state_dict(best_state)
    threshold=.5
    if task=='binary':
        _,val_probs=evaluate(model,x,visual,y,val_i,device,task)
        # Choose threshold on VALIDATION only. Never optimize on test.
        candidates=np.arange(.20,.81,.05)
        threshold=float(max(candidates,key=lambda t:f1_score(y[val_i],val_probs>=t,zero_division=0)))
    val_result,_=evaluate(model,x,visual,y,val_i,device,task,threshold)
    test_result,_=evaluate(model,x,visual,y,test_i,device,task,threshold)
    times=timed_inference(model,x,visual,test_i,device)
    run_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'mode':mode,'task':task,'n_classes':n_classes,
                'channel_count':x.shape[1], 'visual_dim':visual.shape[1],
                'threshold':threshold,'csi_mu':mu,'csi_sd':sd,
                'visual_mu':v_mu,'visual_sd':v_sd},run_dir/f'{mode}.pt')
    return {'mode':mode,'task':task,'n_classes':n_classes,
        'val':val_result,'test':test_result,'test_latency':times,
        'decision_threshold':threshold if task=='binary' else None,
        'train_history':history,'stopped_early_at_epoch':stopped_early_at,
        'checkpoint':str(run_dir/f'{mode}.pt')}


CSV_METRICS=['accuracy','balanced_accuracy','macro_f1','precision','recall',
             'false_positive_rate','brier']


def write_metrics_csv(result, path):
    """One row per model with the held-out TEST metrics (blank when a metric does not apply)."""
    with open(path,'w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['dataset','synthetic_fixture','seed','model',*CSV_METRICS,
                         'latency_p50_ms','latency_p95_ms','test_samples'])
        for mode,model_result in result['models'].items():
            test=model_result['test']
            writer.writerow([result['dataset'],result['synthetic_fixture'],result['seed'],mode,
                *[test.get(metric,'') for metric in CSV_METRICS],
                model_result['test_latency']['p50_ms'],model_result['test_latency']['p95_ms'],
                result['cohort']['test']])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,default=Path('runs/experiment'))
    p.add_argument('--mode',choices=['csi','camera','fusion','all'],default='csi')
    p.add_argument('--group-by',choices=['groups','environment'],default='groups')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=3,help='early stopping; 0 disables')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device=torch.device(args.device)
    with np.load(args.data,allow_pickle=False) as f:
        data={key:f[key] for key in f.files}
    modes=['csi','camera','fusion'] if args.mode=='all' else [args.mode]
    if any(m != 'csi' for m in modes) and not bool(np.all(data['has_visual'])):
        raise ValueError('Camera/fusion requires REAL YOLO video features for every sample. '
                         'Rerun prepare_wimans.py --with-yolo or use --mode csi.')
    if args.group_by not in data:
        raise ValueError(f'{args.group_by} metadata missing')
    split=grouped_split(data[args.group_by],args.seed)
    unique_labels=np.unique(data['y'])
    if len(unique_labels) < 2:
        raise ValueError('Dataset contains only one label; the classification benchmark is undefined.')
    missing = []
    for name, indices in zip(('training','validation','test'),split):
        observed = set(data['y'][indices].tolist())
        if observed != set(unique_labels.tolist()):
            missing.append(f'{name}: {sorted(observed)}')
    if missing:
        raise ValueError('Class coverage missing in grouped split (' + '; '.join(missing) +
            '). Use a larger sample, a better verified grouping, or a different seed. '
            'Do not report this run as an unbiased benchmark.')
    group_sets=[set(data[args.group_by][indices].tolist()) for indices in split]
    ids=data['ids'].astype(str)
    result={'dataset':str(data['dataset'].item()),
        'synthetic_fixture':bool(data['synthetic'].item()),
        'source_npz':str(args.data),
        'source_file_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'seed':args.seed,'group_by':args.group_by,'device':str(device),
        'cohort':{'total':len(data['y']),
           'train':len(split[0]),'validation':len(split[1]),'test':len(split[2]),
           'group_counts':[len(x) for x in group_sets],
           'overlap_groups':len(group_sets[0]&group_sets[1])+
             len(group_sets[0]&group_sets[2])+len(group_sets[1]&group_sets[2]),
           'class_counts':[np.unique(data['y'][i],return_counts=True)[1].tolist() for i in split]},
        'models':{}}
    for mode in modes:
        print(f'Training {mode} on {len(split[0])} samples')
        result['models'][mode]=run_mode(mode,data,args.seed,args.epochs,args.batch_size,
                                       args.lr,device,args.run_dir,split,args.patience)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_metrics_csv(result,args.run_dir/'metrics.csv')
    np.savez(args.run_dir/'heldout_indices.npz',train=split[0],validation=split[1],test=split[2],
             train_ids=ids[split[0]],val_ids=ids[split[1]],test_ids=ids[split[2]])
    print(json.dumps({'real_dataset':not result['synthetic_fixture'],
                      'metrics_path':str(args.run_dir/'metrics.json'),
                      'test_metrics':{m:r['test'] for m,r in result['models'].items()}},indent=2))


if __name__=='__main__':
    main()
