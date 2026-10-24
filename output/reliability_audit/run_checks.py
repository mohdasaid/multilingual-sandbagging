"""Independent cached-activation audit; does not import project analysis helpers.

Seeds 2026..2045, all reported without choosing the best. Intervals conditional
on the original selection; repeated splits are descriptive sensitivity only.
"""
import io, json, zipfile, warnings, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('error', category=ConvergenceWarning)
ROOT=Path('/home/muhammad/Documents/Muhd/Project/provenance-repe')
OUT=Path(__file__).resolve().parent
TABLES=OUT.parent/'corrected_model_reruns'
FILES={'E2B':'E2B_complete_rerun.zip','E4B':'E4B_complete_rerun.zip',
       'AfroLlama':'afrolama_complete_rerun.zip','12B':'12B_complete_rerun.zip'}

def halves(subjects,seed,n=100):
    u=np.unique(subjects); rng=np.random.default_rng(seed)
    return np.array([np.isin(subjects,rng.permutation(u)[:len(u)//2]) for _ in range(n)])

def cosine(a,b):
    return np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))

def rel(x,masks):
    # Matrix multiplication of normalized membership weights: independent of
    # the original implementation's loop over weighted means and unit vectors.
    a=masks.astype(float); b=1-a
    a/=a.sum(1,keepdims=True); b/=b.sum(1,keepdims=True)
    return cosine(a@x,b@x).mean()

def ci(point,deleted):
    deleted=np.asarray(deleted); n=len(deleted)
    se=np.std(deleted,ddof=0)*np.sqrt(n-1)
    return point+np.array([-1,1])*t.ppf(.975,n-1)*se

def fitted(p,n,mask):
    x=np.concatenate((p[mask],n[mask])).astype(float)
    y=np.repeat([1,0],mask.sum())
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=3000,random_state=2026).fit(scaler.transform(x),y)
    return scaler,model

def weight_rel(p,n,masks):
    values=[]
    for mask in masks:
        directions=[]
        for subset in (mask,~mask):
            scaler,model=fitted(p,n,subset)
            directions.append(model.coef_[0]/scaler.scale_)
        values.append(cosine(*directions))
    return float(np.mean(values))

def selection_stats(p,n,lengths,subjects,seed):
    u=np.unique(subjects); chosen=np.random.default_rng(seed).permutation(u)[:10]
    select=np.isin(subjects,chosen); evaluate=~select
    # Match saved float32 subtraction, then accumulate independently in float64.
    delta=(p-n)[select].astype(float)
    masks=halves(subjects[select],seed+1)
    a=masks.astype(float); b=1-a
    a/=a.sum(1,keepdims=True); b/=b.sum(1,keepdims=True)
    flat=delta.reshape(len(delta),-1)
    first=(a@flat).reshape(len(a),*delta.shape[1:])
    second=(b@flat).reshape(len(b),*delta.shape[1:])
    reliability=cosine(first,second).mean(0)
    # Reproduce selection arithmetic; accumulation-rounding tolerance audited.
    mid=(p[select]+n[select])/2
    long=lengths[select]>np.median(lengths[select])
    overlap=np.abs(cosine(mid[long].mean(0)-mid[~long].mean(0),(p[select]-n[select]).mean(0)))
    return select,evaluate,reliability,overlap

def main():
    audit=[]; robustness=[]; matched=[]
    for model,filename in FILES.items():
        baseline=pd.read_csv(TABLES/model/'native.csv')
        layers=pd.read_csv(TABLES/model/'selection_layers.csv')
        with zipfile.ZipFile(ROOT/'rerun tesults'/filename) as z:
            for language in ['eng','hau','yor']:
                for pool in ['last','mean']:
                    name=next(x for x in z.namelist() if x.endswith(f'/{language}/A/{pool}.npz'))
                    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as archive:
                        p=archive['pos']; n=archive['neg']; subjects=archive['subject']; lengths=archive['lengths']
                        assert len(np.unique(archive['pair_id']))==len(p)==100
                        assert np.isfinite(p).all() and np.isfinite(n).all()
                    key=dict(model=model,language=language,pool=pool)
                    base=baseline[(baseline.language==language)&(baseline.pool==pool)&(baseline.threshold==.15)].iloc[0]
                    for seed in range(2026,2046):
                        select,evaluate,floors,overlaps=selection_stats(p,n,lengths,subjects,seed)
                        masks=halves(subjects[evaluate],seed+2)
                        if seed==2026:
                            saved=layers[(layers.language==language)&(layers.pool==pool)].sort_values('layer')
                            ferr=float(np.max(np.abs(saved.selection_floor-floors)))
                            oerr=float(np.max(np.abs(saved.selection_length_overlap-overlaps)))
                            assert ferr<1e-6 and oerr<1e-5,(key,ferr,oerr)
                        for threshold in [.1,.15,.2,.25]:
                            eligible=np.flatnonzero(np.isfinite(floors)&(overlaps<threshold))
                            layer=int(eligible[np.argmax(floors[eligible])]) if len(eligible) else None
                            value=rel((p-n)[evaluate,layer].astype(float),masks) if layer is not None else np.nan
                            robustness.append(dict(key,seed=seed,threshold=threshold,layer=layer,reliability=value))
                            if seed==2026:
                                row=baseline[(baseline.language==language)&(baseline.pool==pool)&(baseline.threshold==threshold)].iloc[0]
                                assert (layer is None)==(row.status=='no_eligible_layer')
                                if layer is not None:
                                    assert layer==int(row.layer)
                                    x=(p-n)[evaluate,layer].astype(float); s=subjects[evaluate]
                                    deleted=[rel(x[s!=v],masks[:,s!=v]) for v in np.unique(s)]
                                    lo,hi=np.clip(ci(value,deleted),-1,1)
                                    error=max(abs(value-row.reliability),abs(lo-row.reliability_ci_lo),abs(hi-row.reliability_ci_hi))
                                    assert error<1e-7,(key,error)
                                    audit.append(dict(key,threshold=threshold,max_geometry_error=error,selection_floor_error=ferr,overlap_error=oerr))
                        if seed==2026:
                            original_select,original_eval=select,evaluate
                    if base.status=='ok':
                        layer=int(base.layer); select=original_select; evaluate=original_eval
                        pp=p[evaluate,layer]; nn=n[evaluate,layer]; s=subjects[evaluate]
                        x=(pp-nn).astype(float); masks=halves(s,2028,n=20)
                        dom=rel(x,masks); probe=weight_rel(pp,nn,masks)
                        assert abs(probe-base.probe_weight_split_half)<1e-5,(key,probe,base.probe_weight_split_half)
                        # Retrain both halves after deleting each complete subject;
                        # uncertainty for the paired difference, not split SD / sqrt(20).
                        deletions=[]
                        for v in np.unique(s):
                            keep=s!=v; reduced=masks[:,keep]
                            deletions.append(rel(x[keep],reduced)-weight_rel(pp[keep],nn[keep],reduced))
                        lo,hi=np.clip(ci(dom-probe,deletions),-2,2)
                        scaler,clf=fitted(p[:,layer],n[:,layer],select)
                        test=np.concatenate((p[evaluate,layer],n[evaluate,layer])).astype(float)
                        accuracy=np.mean(clf.predict(scaler.transform(test))==np.repeat([1,0],evaluate.sum()))
                        assert abs(accuracy-base.same_layer_probe_accuracy)<1e-10
                        matched.append(dict(key,layer=layer,splits=20,dom_reliability=dom,probe_weight_reliability=probe,
                                            difference=dom-probe,ci_lo=lo,ci_hi=hi,same_layer_accuracy=accuracy))
                    for name,rows in [('independent_audit',audit),('repeated_selection',robustness),('matched_estimators',matched)]:
                        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
                    print(f'Completed {model}/{language}/{pool}',flush=True)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(audited_rows=len(audit),robustness_rows=len(robustness),matched_rows=len(matched),
        seeds=list(range(2026,2046)),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

if __name__=='__main__': main()
