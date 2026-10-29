"""Discrete paired inference preserves zero evidence; no positive-only bootstrap weights."""
import itertools,math
import numpy as np
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;a=1+z*z/n;b=(p+z*z/(2*n))/a;c=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [max(0,b-c),min(1,b+c)]
def cluster_paired_difference(a,b,groups,reps=10000):
    d=np.asarray(b,dtype=float)-np.asarray(a,dtype=float);keys=sorted(set(groups));blocks=[d[np.array(groups)==g] for g in keys];rng=np.random.default_rng(20261029);v=[]
    for _ in range(reps):
        draw=rng.integers(0,len(blocks),size=len(blocks));num=sum(blocks[j].sum() for j in draw);den=sum(len(blocks[j]) for j in draw);v.append(num/den)
    return {'point':float(d.mean()),'CI95':np.quantile(v,[.025,.975]).tolist(),'units':len(d),'unit_components':len(keys),'method':'paired whole-unit-component bootstrap allowing omitted discordant components'}
def did_bounds(flags):
    assert len(flags)==4 and all(x in [0,1,2] for x in flags)
    low=high=0
    for flag,sgn in zip(flags,[1,-1,-1,1]):
        low+=sgn*(0 if flag==2 and sgn>0 else 1 if flag==2 else flag)
        high+=sgn*(1 if flag==2 and sgn>0 else 0 if flag==2 else flag)
    return low,high
def paired_difference(a,b,seed=20261029,reps=10000):
    d=np.asarray(b,dtype=float)-np.asarray(a,dtype=float);assert len(d)>0;rng=np.random.default_rng(seed);sample=[]
    for j in range(0,reps,200):
        ix=rng.integers(0,len(d),size=(min(200,reps-j),len(d)));sample.extend(d[ix].mean(axis=1))
    return {'point':float(d.mean()),'CI95':np.quantile(sample,[.025,.975]).tolist(),'units':len(d),'method':'ordinary paired unit bootstrap allowing omitted discordant units'}
def signflip(d):
    d=np.asarray(d,dtype=float);d=d[d!=0];k=len(d)
    if not k:return {'p_two_sided':1.,'nonzero_units':0,'method':'exact'}
    target=abs(d.sum());n=0;hit=0
    if k<=20:
        for signs in itertools.product([-1,1],repeat=k):n+=1;hit+=abs(np.dot(signs,d))>=target-1e-12
        return {'p_two_sided':hit/n,'nonzero_units':k,'method':'exact paired unit sign-flip','permutations':n}
    rng=np.random.default_rng(20261029);n=100000
    for j in range(0,n,1000):hit+=int(np.sum(np.abs((rng.integers(0,2,size=(1000,k))*2-1)@d)>=target-1e-12))
    return {'p_two_sided':(hit+1)/(n+1),'nonzero_units':k,'method':'100000 fixed-seed paired unit sign-flips with plus-one correction','permutations':n}
def two_way_difference(a,b,units,sources,reps=10000):
    d=np.asarray(b,dtype=float)-np.asarray(a,dtype=float);u=sorted(set(units));s=sorted(set(sources));ui=np.array([u.index(x) for x in units]);si=np.array([s.index(x) for x in sources]);rng=np.random.default_rng(20261029);out=[];redraw=0
    while len(out)<reps:
        uw=rng.multinomial(len(u),np.full(len(u),1/len(u)));sw=rng.multinomial(len(s),np.full(len(s),1/len(s)));w=uw[ui]*sw[si]
        if w.sum()==0:redraw+=1;continue
        out.append(float(np.dot(w,d)/w.sum()))
    return {'point':float(d.mean()),'CI95':np.quantile(out,[.025,.975]).tolist(),'units':len(u),'source_components':len(s),'redrawn_zero_denominators':redraw,'method':'paired two-way multinomial pigeonhole bootstrap; observed discordances can be absent'}
