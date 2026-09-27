import numpy as np
# Sharpen (iii): the iff says "learned-absolute solves iff fixed in-range position".
# But SINUSOIDAL is *absolute* (kernel = function of (i,j) via fixed features) AND defined everywhere,
# AND it can act relatively (cos(ang_i-ang_j)). So is sinusoidal "absolute" for the iff or not?
# Test: does sinusoidal SOLVE a fixed-OFFSET task (prev-token) under min-norm selection?  -> the iff
# for learned-absolute would predict CHANCE. But sinusoidal is NOT learned-absolute (Thm 2b-LA needs
# RANDOM untrained features). So Thm 2b-LA does NOT apply. Selection can still fail or succeed.

def sinusoidal(n, m, base=10000.0):
    pos=np.arange(n)[:,None]; i=np.arange(m//2)[None,:]
    ang=pos/np.power(base,2*i/m)
    Phi=np.zeros((n,m)); Phi[:,0::2]=np.sin(ang); Phi[:,1::2]=np.cos(ang); return Phi

def solve_prev(L0,L,m,base=10000.0):
    Phi0=sinusoidal(L0,m,base); PhiL=sinusoidal(L,m,base)
    R0=np.zeros((L0,L0))
    for i in range(1,L0): R0[i,i-1]=1.0
    W=np.linalg.pinv(Phi0)@R0@np.linalg.pinv(Phi0).T
    PL=PhiL@W@PhiL.T
    c=0;t=0
    for i in range(L0,L):
        if int(np.argmax(PL[i,:i+1]))==i-1: c+=1
        t+=1
    return c/t

# vary m (resolution). Higher m => more frequencies => better relative extrapolation?
print("sinusoidal min-norm selection on prev-token (fixed offset), new-row acc:")
for m in [8,16,32,64,128]:
    print(f"  m={m:4d}: acc={solve_prev(20,80,m):.3f}")

# Now: is the FAILURE 'chance' (=1/k like learned-abs) or structured?
# Compare to the learned-abs chance baseline 1/k averaged over new rows:
ks=[ (i+1) for i in range(20,80) ]
chance=np.mean([1.0/k for k in ks])
print(f"\nlearned-abs chance baseline (avg 1/(i+1)) = {chance:.4f}")
print("=> if sinusoidal acc is well ABOVE chance, the absolute-fails-fixed-offset iff direction")
print("   does NOT hold for deterministic sinusoidal; it is a SELECTION (not chance) phenomenon.")

# Direct: EXISTENCE of W solving prev-token on new rows for sinusoidal (relative cos kernel shifted)
# Build W implementing a peak at offset=1: use weighting so sum_f cos(ang_i-ang_j) peaks near d=1.
# Simpler: search a diagonal W (per-frequency gains) to peak the relative kernel at d=1.
m=64; PhiL=sinusoidal(80,m)
# relative kernel value at offset d with diagonal W=diag(w): K(d)=sum_f w_f cos(d*omega_f) (pairing sin,cos)
def relkernel_acc(wdiag):
    W=np.diag(wdiag); PL=PhiL@W@PhiL.T
    c=0;t=0
    for i in range(20,80):
        if int(np.argmax(PL[i,:i+1]))==i-1: c+=1
        t+=1
    return c/t
# try all-ones (W=I) shifted: actually want peak at d=1 not d=0. Use phase: impossible with diag on this basis simply.
print(f"\nW=I existence (peak d=0, target=i): would solve 'copy-self' offset 0, acc for prev=", relkernel_acc(np.ones(m)))
