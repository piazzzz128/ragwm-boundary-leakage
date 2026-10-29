from contracts import canon,digest,request_key,positions

def flag(text):
    if text in ['Yes','YES','yes','Yes.','YES.','yes.']:return 1
    if text in ['No','NO','no','No.','NO.','no.']:return 0
    return 2

def payload_indices(direct,simulated):
    if not direct or len(direct)!=len(simulated):raise ValueError('invalid generation bundle')
    both=[i for i,(a,b) in enumerate(zip(direct,simulated)) if a==b==1]
    if both:return both
    d=[i for i,x in enumerate(direct) if x==1]
    if d:return d[:1]
    d=[i for i,x in enumerate(simulated) if x==1]
    return d[:1] if d else [0]
