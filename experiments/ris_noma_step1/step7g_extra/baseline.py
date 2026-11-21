"""Train-only polynomial ridge baseline; no graph features or audit information."""
import numpy as np
from .common import allocation_errors
from ris_noma.multik_core import physical_allocation


def features(row, rank=None):
    # dBm is already logarithmic power. No log(dBm), coordinates or gain inputs.
    x,y,k=(row["xi"]-.85)/.10,(row["power_dbm"]-35.)/5.,(row["n_public"]-3.5)/1.5
    f=[x,y,k,x*x,y*y,k*k,x*y,x*k,y*k]
    if rank is not None:
        r=(rank+.5)/row["n_public"]
        f += [r,r*r,r*x,r*y,r*k]
    return f


def regress(x,y,alpha):
    x,y=np.asarray(x,float),np.asarray(y,float)
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-12,1.,scale)
    z=np.c_[np.ones(len(x)),(x-mean)/scale]
    penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    coef=np.linalg.solve(z.T@z+penalty,z.T@y)
    return {"mean":mean.tolist(),"scale":scale.tolist(),"coef":coef.tolist()}


def value(f,record):
    return np.r_[1.,(np.asarray(f)-record["mean"])/record["scale"]]@np.asarray(record["coef"])


def fit(rows,bounds,alpha):
    if not rows or any(r["split"]!="train" or not r["has_label"] for r in rows):
        raise ValueError("Baseline can fit labeled train rows only")
    lo=np.array([bounds["mb"][0],bounds["s"][0]]);hi=np.array([bounds["mb"][1],bounds["s"][1]])
    targets=np.asarray([[r["allocation"][0],r["allocation"][-1]] for r in rows])
    unit=np.clip((targets-lo)/(hi-lo),1e-6,1-1e-6)
    public_x=[];public_y=[]
    for r in rows:
        y=np.asarray(r["allocation"]);p=y[1:-1][r["order"]]/(1-y[0]);logp=np.log(np.maximum(p,1e-12));logp-=logp.mean()
        for rank,target in enumerate(logp):public_x.append(features(r,rank));public_y.append(target)
    return {"alpha":alpha,"fit_role":"train","train_layout_ids":sorted({r["layout_id"] for r in rows}),
        "training_cases":len(rows),"inputs":["xi","power_dbm","n_public","public_SIC_rank"],"bounds":bounds,
        "scalars":regress([features(r) for r in rows],np.log(unit/(1-unit)),alpha),
        "public":regress(public_x,public_y,alpha),"public_policy":"centered log share by normalized SIC rank; softmax and inverse order"}


def predict(record,rows):
    bounds=record["bounds"];result=[]
    for r in rows:
        scalars=value(features(r),record["scalars"]);u=1/(1+np.exp(-np.clip(scalars,-40,40)))
        mb=bounds["mb"][0]+np.ptp(bounds["mb"])*u[0];s=bounds["s"][0]+np.ptp(bounds["s"])*u[1]
        scores=np.asarray([value(features(r,j),record["public"]) for j in range(r["n_public"])])
        share=np.exp(scores-scores.max());share/=share.sum()
        y=np.r_[mb,np.zeros(r["n_public"]),s];y[1:-1][r["order"]]=(1-mb)*share
        result.append(physical_allocation(y,r["n_public"],bounds))
    return result


def lower_bob(y,delta,bounds):
    if not np.isfinite(delta) or delta<0:raise ValueError("Delta must be finite and nonnegative")
    y=physical_allocation(y,len(y)-2,bounds)
    old=y[0];y[0]=max(bounds["mb"][0],old-delta)
    y[1:-1]*=(1-y[0])/(1-old)
    return physical_allocation(y,len(y)-2,bounds)
