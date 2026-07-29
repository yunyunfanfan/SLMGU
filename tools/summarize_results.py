import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root', nargs='+', default=['outputs/checkpoints'])
args=p.parse_args()
rows=[]
for root in args.root:
    for f in Path(root).rglob('trainer_log.json'):
        try:
            d=json.load(open(f))
        except Exception:
            continue
        if 'dt_auc' in d:
            rows.append((str(f), d.get('dt_auc'), d.get('dt_aup'), d.get('df_auc'), d.get('df_aup')))
for r in rows:
    print('	'.join(map(str,r)))
if rows:
    arr=np.array([[x if isinstance(x,(int,float)) else np.nan for x in r[1:]] for r in rows], dtype=float)
    print('mean dt_auc dt_aup df_auc df_aup:', np.nanmean(arr, axis=0).tolist())
