"""Deterministic training-only 17-field profile."""
import csv, math
from pathlib import Path
from statistics import median
from datetime import datetime, timezone

FIELDS=("rolling_location_max","rolling_location_median","SB_BinaryStats_diff_longstretch0","kpss_level_auto_statistic","train_eval_median_shift","excess_kurtosis","permutation_entropy","number_peaks","B01","A03","A04","C01")
MISSING={"","nan","na","null","none","n/a"}
def num(v):
    if v is None or v.strip().lower() in MISSING:return None
    try:
        x=float(v); return x if math.isfinite(x) else None
    except ValueError:return None
def med(a): return median(a) if a else None
def mad(a):
    m=med(a); return med([abs(x-m) for x in a]) if m is not None else None
def interp(a):
    if not a or sum(x is None for x in a)/len(a)>.05:return None
    out=list(a); ix=[i for i,x in enumerate(out) if x is not None]
    if not ix:return None
    for i in range(len(out)):
        if out[i] is not None:continue
        l=max((j for j in ix if j<i),default=None); r=min((j for j in ix if j>i),default=None)
        out[i]=out[l] if r is None else out[r] if l is None else out[l]+(out[r]-out[l])*(i-l)/(r-l)
    return out
def zscore(a):
    m=sum(a)/len(a); s=math.sqrt(sum((x-m)**2 for x in a)/len(a))
    return ([((x-m)/s) for x in a],s) if s and math.isfinite(s) else (None,s)
def events(y):
    out=[];i=0
    while i<len(y):
        if y[i]!=1:i+=1;continue
        j=i+1
        while j<len(y) and y[j]==1:j+=1
        out.append((i,j));i=j
    return out
def timestamp(v):
    try:return float(v)
    except (TypeError,ValueError):
        try:return datetime.fromisoformat(v.replace('Z','+00:00')).replace(tzinfo=timezone.utc if '+' not in v else None).timestamp()
        except Exception:return None
def training_profile(path:Path)->dict:
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); cols=r.fieldnames or []; rows=list(r)
    if not cols or len(set(cols))!=len(cols):raise ValueError('Training CSV requires unique column names.')
    nums={c:[num(row.get(c)) for row in rows] for c in cols if c.lower() not in {'timestamp','label'}}
    nums={c:v for c,v in nums.items() if any(x is not None for x in v)}; name=next(iter(nums),None); raw=nums.get(name,[]); x=interp(raw)
    p={k:None for k in FIELDS}; status={k:'unavailable' for k in FIELDS}
    if x is not None:
        z,s=zscore(x)
        if z is not None:
            w=max(32,round(.05*len(z))); ws=[z[i:i+w] for i in range(0,len(z)-w+1,w)]; ds=[abs(med(a)-med(b)) for a,b in zip(ws,ws[1:])]
            if len(ws)>=4:p['rolling_location_max']=max(ds);p['rolling_location_median']=med(ds);status['rolling_location_max']=status['rolling_location_median']='ok'
            p['number_peaks']=sum(all(z[i]>z[i-j] and z[i]>z[i+j] for j in range(1,6)) for i in range(5,len(z)-5))/len(z);status['number_peaks']='ok'
            if len(z)>=32:
                counts={}
                for i in range(len(z)-2):
                    q=tuple(sorted(range(3),key=lambda j:(z[i+j],j)));counts[q]=counts.get(q,0)+1
                n=sum(counts.values());p['permutation_entropy']=-sum(v/n*math.log(v/n) for v in counts.values());status['permutation_entropy']='ok'
            try:
                from statsmodels.tsa.stattools import kpss;p['kpss_level_auto_statistic']=float(kpss(z,regression='c',nlags='auto')[0]);status['kpss_level_auto_statistic']='ok'
            except Exception:status['kpss_level_auto_statistic']='numerical_error'
        try:
            from scipy.stats import kurtosis;p['excess_kurtosis']=float(kurtosis(x,fisher=True,bias=False));status['excess_kurtosis']='ok'
        except Exception:status['excess_kurtosis']='dependency_unavailable'
        try:
            import pycatch22; rr=pycatch22.catch22_all(x,catch24=False,short_names=False);p['SB_BinaryStats_diff_longstretch0']=dict(zip(rr['names'],rr['values'])).get('SB_BinaryStats_diff_longstretch0');status['SB_BinaryStats_diff_longstretch0']='ok'
        except Exception:status['SB_BinaryStats_diff_longstretch0']='dependency_unavailable'
        status['train_eval_median_shift']='missing_eval'
    if 'label' in cols and all(num(row.get('label')) in (0,1) for row in rows):
        y=[int(num(row.get('label'))) for row in rows]; ev=events(y);p['A03']=1000*len(ev)/len(y) if y else None;p['A04']=med([b-a for a,b in ev]) if ev else None;status['A03']='ok';status['A04']='ok' if ev else 'no_event'
    n=len(rows); nf=len(nums); missing=sum(v is None for values in nums.values() for v in values)
    constants=sum(bool([v for v in values if v is not None]) and len(set(v for v in values if v is not None))==1 for values in nums.values())
    regular=None
    if 'timestamp' in cols and n>=3:
        ts=[timestamp(row.get('timestamp')) for row in rows]
        if all(v is not None for v in ts):
            gaps=[b-a for a,b in zip(ts,ts[1:])]; regular=all(g>0 and math.isclose(g,gaps[0],rel_tol=1e-6,abs_tol=1e-9) for g in gaps)
    return {**p,'training_rows':n,'numeric_features':nf,'missing_ratio':missing/(n*nf) if n and nf else None,'constant_column_ratio':constants/nf if nf else None,'regular_sampling':regular,'profile_version':'training-profile-v2-17-fields','profile_signal':name,'field_status':status,'definition':'Training-only 17-field profile: five structural fields plus twelve signal/event fields; D1 fields require complete training labels.'}
