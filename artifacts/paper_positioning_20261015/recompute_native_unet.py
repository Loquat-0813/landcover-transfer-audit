"""Reconstruct native U-Net tables from original integer confusion records."""
import hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
B=R/'artifacts/audit_submission_20261004/cloud_evidence/outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs={};arms={};points={};boundaries={};expected_ids=None
for arm in ('unet_b0','unet_b0_half','unet_b1'):
    rows=[]
    for seed in (19,37,73):
        directory=B/f'unet_oem_{arm}_seed{seed}'
        d=json.loads((directory/'city_confusions.json').read_text());names=sorted(d['city_confusions'])
        raw=json.loads((directory/'per_raster_confusions.json').read_text())['confusions']
        assert len(raw)==2638 and len(names)==74 and 'paris' not in names
        if expected_ids is None:expected_ids=sorted(raw)
        assert sorted(raw)==expected_ids
        mats=np.array([d['city_confusions'][c] for c in names],dtype=np.int64)
        for city,m in zip(names,mats):assert np.array_equal(m,np.sum([a for key,a in raw.items() if key.split('/')[0]==city],axis=0))
        rows.append(mats)
        for name in ('city_confusions.json','per_raster_confusions.json','metrics.json','run_config.json'):inputs[str(directory/name)]=sha(directory/name)
    arms[arm]=np.array(rows)
def iou(m):
    tp=np.diagonal(m,axis1=-2,axis2=-1);n=m.sum(-1);u=n+m.sum(-2)-tp
    return np.divide(tp,u,out=np.full(tp.shape,np.nan),where=(n>0)&(u>0))
for arm,m in arms.items():
    v=iou(m.sum(1));p={'rangeland_iou':v[:,3], 'supported_leaf_miou':v[:,[0,1,2,4,5,6]].mean(1),'overall_miou':v.mean(1)}
    points[arm]={name:{'values':vals.tolist(),'mean':float(np.mean(vals)),'sample_sd':float(np.std(vals,ddof=1))} for name,vals in p.items()}
draw=np.random.default_rng(20260803).integers(0,74,(2000,74))
boots={a:np.mean([iou(m[draw].sum(1)) for m in rows],axis=0) for a,rows in arms.items()}
for base in ('unet_b0','unet_b0_half'):
    delta=boots['unet_b1']-boots[base];rows={}
    for name,ix in [('rangeland_iou',[3]),('supported_leaf_miou',[0,1,2,4,5,6]),('overall_miou',list(range(7)))]:
        rows[name]={'point':points['unet_b1'][name]['mean']-points[base][name]['mean'],'CI':np.quantile(delta[:,ix].mean(1),[.025,.975]).tolist()}
    boundaries['unet_b1 minus '+base]=rows
out={'source':'original native integer records, no scientific re-evaluation','inputs':inputs,'seed_order':[19,37,73],'target_pairs':2638,'cities':74,'bootstrap_seed':20260803,'bootstrap_draws':2000,'points':points,'contrasts':boundaries,'shuffle':'excluded from this analysis; preserved for independent integrity audit'}
(H/'NATIVE_UNET_RECOMPUTED.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'native_points':points,'contrasts':boundaries}))
