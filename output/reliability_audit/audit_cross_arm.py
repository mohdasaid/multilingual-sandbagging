"""Independent audit of every saved cross-arm geometry entry and interval."""
import io,json,zipfile
import numpy as np
import pandas as pd
from run_checks import ROOT,OUT,TABLES,FILES,halves,rel,cosine,ci

def ceiling(a,b):
    return np.sqrt((2*a/(1+a))*(2*b/(1+b))) if 0<=a<=1 and 0<=b<=1 else np.nan

def summarize(archive,layer,subjects):
    keep=np.isin(archive['subject'],subjects)
    s=archive['subject'][keep]
    x=(archive['pos'][:,layer]-archive['neg'][:,layer])[keep].astype(float)
    masks=halves(s,2028)
    point=(rel(x,masks),x.mean(0))
    deleted=[]
    for v in np.unique(s):
        k=s!=v
        deleted.append((rel(x[k],masks[:,k]),x[k].mean(0)))
    return point,deleted

def main():
    results=[]
    for model,filename in FILES.items():
        table=pd.read_csv(TABLES/model/'cross_arm.csv')
        with zipfile.ZipFile(ROOT/'rerun tesults'/filename) as z:
            for language in ['hau','yor']:
                for pool in ['last','mean']:
                    data={}
                    rows=table[(table.language==language)&(table.pool==pool)]
                    for arm in ['A','B','C','D']:
                        name=next(n for n in z.namelist() if n.endswith(f'/{language}/{arm}/{pool}.npz'))
                        with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
                            data[arm]={k:a[k] for k in ['pos','neg','subject']}
                    native=data['A']; unique=np.unique(native['subject'])
                    subjects=np.random.default_rng(2026).permutation(unique)[10:]
                    assert all(set(a['subject'])==set(unique) for a in data.values())
                    cache={}
                    for _,row in rows.iterrows():
                        layer=int(row.layer)
                        for arm in ['A',row.arm]:
                            if (arm,layer) not in cache:
                                cache[arm,layer]=summarize(data[arm],layer,subjects)
                        a,ad=cache['A',layer]; b,bd=cache[row.arm,layer]
                        c=float(cosine(a[1],b[1])); h=ceiling(a[0],b[0])
                        cd=[cosine(x[1],y[1]) for x,y in zip(ad,bd)]
                        hd=[ceiling(x[0],y[0]) for x,y in zip(ad,bd)]
                        clo,chi=np.clip(ci(c,cd),-1,1)
                        hlo,hhi=np.clip(ci(h,hd),0,1)
                        glo,ghi=ci(h-c,np.array(hd)-cd)
                        actual=np.array([c,h,h-c,clo,chi,hlo,hhi,glo,ghi])
                        expected=row[['cosine','ceiling_heuristic','ceiling_minus_cosine','cosine_ci_lo','cosine_ci_hi',
                                      'ceiling_ci_lo','ceiling_ci_hi','gap_ci_lo','gap_ci_hi']].to_numpy(float)
                        assert np.array_equal(np.isnan(actual),np.isnan(expected)),(model,language,pool,row.arm)
                        err=float(np.nanmax(abs(actual-expected)))
                        assert err<1e-7,(model,language,pool,row.arm,err)
                        results.append(dict(model=model,language=language,pool=pool,arm=row.arm,threshold=row.threshold,max_error=err))
                    print(f'Cross-arm audit {model}/{language}/{pool}: passed',flush=True)
                    pd.DataFrame(results).to_csv(OUT/'cross_arm_audit.csv',index=False)
    (OUT/'CROSS_ARM_COMPLETE.json').write_text(json.dumps({'rows_checked':len(results),'max_error':max(r['max_error'] for r in results)},indent=2))

if __name__=='__main__': main()
