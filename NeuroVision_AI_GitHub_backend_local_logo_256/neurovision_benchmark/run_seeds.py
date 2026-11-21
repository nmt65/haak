"""Run isolated seeds and aggregate held-out metrics honestly."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--mode', choices=['csi', 'camera', 'fusion', 'all'], default='all')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--group-by', choices=['groups','environment'], default='groups')
    p.add_argument('--run-dir', type=Path, default=Path('runs/multi_seed'))
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--patience', type=int, default=3)
    args = p.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError('Specify at least two unique seeds.')
    records=[]
    for seed in seeds:
        run_dir = args.run_dir / f'seed_{seed}'
        cmd = [sys.executable,'train.py','--data',str(args.data),'--mode',args.mode,
               '--epochs',str(args.epochs),'--group-by',args.group_by,
               '--seed',str(seed),'--device',args.device,'--run-dir',str(run_dir),
               '--patience',str(args.patience)]
        subprocess.run(cmd, check=True)
        records.append(json.loads((run_dir/'metrics.json').read_text()))
    datasets = {r['dataset'] for r in records}
    synthetic_values = {r['synthetic_fixture'] for r in records}
    if len(datasets)!=1 or len(synthetic_values)!=1:
        raise RuntimeError('Seeds came from inconsistent datasets.')
    modes = sorted(records[0]['models'])
    # Metrics that exist for this task (precision/recall/FPR only exist for binary presence).
    candidate_metrics=['accuracy','macro_f1','balanced_accuracy','precision','recall','false_positive_rate']
    metrics=[m for m in candidate_metrics if m in records[0]['models'][modes[0]]['test']]
    out = {
      'dataset':records[0]['dataset'],
      'synthetic_fixture':records[0]['synthetic_fixture'],
      'seeds':seeds,
      'evaluation':'held-out test of each group-split seed',
      'models':{},
    }
    for mode in modes:
        out['models'][mode]={}
        for metric in metrics:
            vals=[r['models'][mode]['test'][metric] for r in records]
            out['models'][mode][metric]={
              'per_seed':vals,
              'mean':mean(vals),
              'sample_std':stdev(vals),
            }
        latencies=[r['models'][mode]['test_latency']['p50_ms'] for r in records]
        out['models'][mode]['latency_p50_ms']={'per_seed':latencies,'mean':mean(latencies),
                                               'sample_std':stdev(latencies)}
    args.run_dir.mkdir(parents=True,exist_ok=True)
    output=args.run_dir/'aggregate.json'
    output.write_text(json.dumps(out,indent=2), encoding='utf8')
    with open(args.run_dir/'aggregate.csv','w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['dataset','synthetic_fixture','model','metric','mean','sample_std','per_seed'])
        for mode,values in out['models'].items():
            for metric,summary in values.items():
                writer.writerow([out['dataset'],out['synthetic_fixture'],mode,metric,
                    round(summary['mean'],4),round(summary['sample_std'],4),
                    ';'.join(str(round(v,4)) for v in summary['per_seed'])])
    print('Wrote',output, '(SYNTHETIC SMOKE ONLY)' if out['synthetic_fixture'] else '(REAL DATASET EVALUATION)')

if __name__=='__main__':
    main()
