import numpy             as np
import numpy.linalg      as la
import cv2               as cv

import numpy.linalg      as la
import cmath

import umucv.contours    as cnt
from umucv.contours  import eig22, extractContours
from umucv.htrans    import vec, desp, rot3, scale, rotation, kgen
from umucv.shape     import spectralFeat, invar

try:
    from scipy.ndimage   import distance_transform_edt
    from scipy.special   import binom
except ImportError:
    pass


def mkPowers(sz=(500,500),res=100):
    global POWERS
    global RESOL
    RESOL =res
    w,h = sz
    y =  np.outer(range(h),np.ones(w))/res
    x =  np.outer(np.ones(h),range(w))/res
    POWERS = np.array(
        [ np.ones((h,w))
        , x
        , y
        , x*x
        , x*y
        , y*y
    
        , x*x*x
        , x*x*y
        , x*y*y
        , y*y*y
    
        , x*x*x*x
        , x*x*x*y
        , x*x*y*y
        , x*y*y*y
        , y*y*y*y
        ])

mkPowers()


def rawMoments(p, full=True):
    h,w = p.shape
    M = np.zeros([5,5])
    if full:
        m = np.einsum('ij,pij',p,POWERS[:,:h,:w])
        M[0,0], M[1,0],M[0,1], M[2,0],M[1,1],M[0,2], M[3,0],M[2,1],M[1,2],M[0,3], M[4,0],M[3,1],M[2,2],M[1,3],M[0,4] = m
    else:
        m = np.einsum('ij,pij',p,POWERS[:6,:h,:w])
        M[0,0], M[1,0],M[0,1], M[2,0],M[1,1],M[0,2] = m
    return M


def whitener(X, th=10):
    M = rawMoments(X, full=False)
    A = M[0,0]
    M /= A
    mx = M[1,0]
    my = M[0,1]
    sxx = M[2,0] - mx**2
    sxy = M[1,1] - mx*my
    syy = M[0,2] - my**2
    (l1,l2,ad) = eig22(sxx,syy,sxy)
    s1 = np.sqrt(l1); s2 = np.sqrt(l2)
    if abs(s2/s1) < 1/th:
        sxx = l1
        syy = l1
        sxy = 0
    a = np.sqrt(syy/(sxx*syy-sxy**2))
    c = 1/np.sqrt(syy)
    b = -a*sxy/syy
    e = -a*mx - b*my
    f = -c*my
    H = np.array([[a,b,e],[0,c,f],[0,0,1]])          
    return (mx,my,sxx,sxy,syy),(A,s1,s2,ad),H




def whitenedMoments(X, th=10):
    M = rawMoments(X)
    n = 0
    A = M[0,0]
    M /= A
    mx = M[1,0]
    my = M[0,1]
    C = np.zeros([5,5])
    for p in range(5):
        for q in range(5):
            if p+q <= 4:
                #print(p,q)
                for j in range(p+1):
                    for k in range(q+1):
                        #print('   ',j,k)
                        C[p,q] += binom(p,j)*binom(q,k)*M[j,k]*(-mx)**(p-j)*(-my)**(q-k)
                        n += 1
    #print('cen: ',n)
    n = 0
    W = np.zeros([5,5])
    sxx = C[2,0]
    sxy = C[1,1]
    syy = C[0,2]
    (l1,l2,ad) = eig22(sxx,syy,sxy)
    s1 = np.sqrt(l1); s2 = np.sqrt(l2)
    #print(s1,s2,ad)
    if abs(s2/s1) < 1/th:
        #print('dege ', abs(s2/s1), th)
        sxx = l1
        syy = l1
        sxy = 0
    a = np.sqrt(syy/(sxx*syy-sxy**2))
    c = 1/np.sqrt(syy)
    b = -a*sxy/syy
    for p in range(5):
        for q in range(5):
            if p+q <= 4:
                #print(p,q)
                for j in range(p+1):
                    #print('   ',j)
                    W[p,q] += binom(p,j)*a**j*b**(p-j)*c**q *  C[j,p+q-j]
                    n += 1
    #print('whi: ',n)
    e = -a*mx - b*my
    f = -c*my
    H = np.array([[a,b,e],[0,c,f],[0,0,1]])          
    return (A,s1,s2,ad),H,W



def mkAngles():
    a = np.radians(np.arange(360))
    c = np.cos(a)
    s = np.sin(a)
    return [ a
           , c**3
           , c**2*s
           , c*s**2
           , s**3
           , c**4
           , c**3*s
           , c**2*s**2
           , c*s**3
           , s**4
           ]

CS = mkAngles()


def KSsignature(W):
    T = CS
    
    sk = W[3,0]*T[1] + 3*W[2,1]*T[2] + 3*W[1,2]*T[3] + W[0,3]*T[4]
    ku = W[4,0]*T[5] + 4*W[3,1]*T[6] + 6*W[2,2]*T[7] + 4*W[1,3]*T[8] + W[0,4]*T[9]
    sb = 3/8*(W[3,0]+W[1,2]) , 3/8*(W[2,1]+W[0,3])
    
    return sb,sk,ku




def localmax(v):
    va = np.roll(v,1)
    vs = np.roll(v,-1)
    mx = v.max()
    return np.where(np.logical_and(v>va,v>vs))[0]


def orientability(SK):
        SBar,S,K = SK
        r = K.max() - K.min()
        s = la.norm(SBar)
        
        ka = localmax(-K)
        if len(ka) == 4:
            ka = sorted(ka,key=lambda x: K[x])
            t = K[ka[-1]] - K[ka[0]]
        
            kb = localmax(K)
            kb = sorted(kb,key=lambda x: K[x])
            g  = K[kb[0]] - K[ka[-1]]
        else:
            t = 0
            g = 0
        return r,s,t,g


def diff(v):
    va = np.roll(v,1)
    return v-va



def deltaAngle(a,b):
    d = np.exp(np.radians(a)*1j) / np.exp(np.radians(b)*1j)
    return np.degrees(cmath.phase(d))


def canonicAngles(SK, eps_r=0.2, eps_t=0.75, eps_s=0.04, delta=20, offset=10, debug=False):
    SBar,K = SK
    
    r = K.max() - K.min()
    if r < eps_r:
        return [0]
    
    ka = localmax(-K)
    ka = sorted(ka,key=lambda x: K[x])
    
    if debug: print(ka); print(K[ka])
    
    t = K[ka[-1]] - K[ka[0]]
    
    if 1-t/r < eps_t:
        ka = ka[:2]
    
    ka = sorted(ka)
       
    if debug: print(ka); print(K[ka])

    
    s = la.norm(SBar)
    if debug: print('||S||:',s)
    if s > eps_s:
        alpha = np.degrees(np.arctan2(SBar[1],SBar[0]))
        if debug: print('alpha:',alpha)
        del2 = delta/2
        cens = [(k, ka[k] + offset) for k in range(len(ka))]
        d1 = [abs(deltaAngle(alpha,c+del2)) for _,c in cens]
        d2 = np.argmax([np.cos(np.radians(alpha-c-del2)) for _,c in cens])
        sel = [[ka[k],ka[(k+1) % len(ka)]] for k,c in cens if abs(deltaAngle(alpha,c+del2)) < del2]
        
        if debug:
            print(cens)
            print(d1,d2)
            print(sel)
        
        sel = sum(sel,[])
        
        if not sel:
            d3 = [(k, deltaAngle(c,alpha)) for k,c in cens]
            sel= [ka[min([(k,a) for k,a in d3 if a>0], key=lambda x: x[1])[0]]]
            
            if debug:
                print(d3)
        
    else:
        sel = ka

    if debug: print(sel)
    
    angs = np.radians(sorted(sel,key=lambda x: K[x]))
    
    return angs

def canonicTransforms(SK, **args):
    angs = canonicAngles(SK,**args)
    return [rot3(-a) for a in angs]


#for python 2
def dots(*args):
    r = args[0]
    for x in args[1:]:
        r = np.dot(r,x)
    return r


def warpW(img, H, sz=(100,100), sigma=3, offset=0, **args):
    w,h = sz
    w2 = w/2
    h2 = h/2
    #return cv.warpPerspective(img, desp([w2,h2]) @ np.diag([1,1,sigma/w2]) @ H @ np.diag([1,1,RESOL]) @ desp([-offset,-offset]),sz, **args)
    T = dots(desp([w2,h2]) , np.diag([1,1,sigma/w2]) , H , np.diag([1,1,RESOL]) , desp([-offset,-offset]))
    return cv.warpPerspective(img, T ,sz, **args)

def whiten(X, sz=(100,100), sigma=3, th=10):
    _,_,H = whitener(X, th=th)
    return warpW(X, H, sz=sz, sigma=sigma)

def normalize(X, sz=(100,100), sigma=3):
    return whiten(X, sz=sz, sigma=sigma, th=1/2)


def canonicalize(X, sz=(100,100), sigma=3, eps_w=10, debug=False, **args):
    _, H, W    = whitenedMoments(X, th=eps_w)
    SBar, S, K = KSsignature(W)
    Ts         = [ np.dot(t , H) for t in canonicTransforms((SBar,K), debug=debug, **args) ]
    can        = [ warpW(X, T, sz=sz, sigma=sigma) for T in Ts ]
    return can


def dt(x):
    return distance_transform_edt(x<0.5)

def sdt(x):
    return distance_transform_edt(x<0.5) - distance_transform_edt(x>=0.5)


class Model:
    def __init__(self, x, lab='?', full=False, eps_w=10, **args):
        self.orig  = x
        (A, self.s1, self.s2, ad), H, W = whitenedMoments(x, th=eps_w)
        self.H = H
        self.W = W

        self.SBar, self.S, self.K   = KSsignature(W)

        self.Ts    = [ np.dot(t , H) for t in canonicTransforms((self.SBar,self.K), **args) ]
        self.can   = [ warpW(x, T, sz=(100,100), sigma=3) for T in self.Ts ]
        self.dtc   = [ dt(c) for c in self.can ]
        self.auxh1 = [ np.hstack([self.can[k],self.dtc[k]]) for k in range(len(self.can)) ] 
        self.auxh2 = [ np.hstack([self.dtc[k],self.can[k]]) for k in range(len(self.can)) ] 
        
        pl         = min(x.shape)
        self.pad   = np.pad(x, pl, 'constant')

        if full:
            self.dto   = dt(self.pad) / self.s1
            self.dtco  = [ warpW(self.dto, T, sz=(100,100), sigma=3, offset=pl, borderMode=cv.BORDER_REPLICATE) for T in self.Ts ]
            self.auxh3 = [ np.hstack([self.can[k],self.dtco[k]]) for k in range(len(self.can)) ] 
            self.auxh4 = [ np.hstack([self.dtco[k],self.can[k]]) for k in range(len(self.can)) ]

        conts = extractContours(255-255*self.pad.astype(np.uint8),minarea=0,minredon=0,reduprec=0.5)
        assert len(conts)>0, (len(conts),lab)
        self.wcont = sorted(conts,key=len)[-1]
        self.invar = invar(spectralFeat(self.wcont))

        r,s,t,g = orientability((self.SBar, self.S, self.K))
        self.skf = vec(self.S.max(),
                       s,
                       (self.K.min() + self.K.max())/2,
                       r,
                       t/r,
                       g/r)
        self.lab   = lab

##########################################################################


def fixH(H, sz=(100,100), sigma=3):
    w,h = sz
    w2 = w/2
    h2 = h/2
    #return desp([w2,h2]) @ np.diag([1,1,sigma/w2]) @ H @ np.diag([1,1,RESOL])
    return dots(desp([w2,h2]) , np.diag([1,1,sigma/w2]) , H , np.diag([1,1,RESOL]))

def alignments(ma,mb):
    return [np.dot(la.inv(fixH(Ta)) , fixH(Tb)) for Ta in ma.Ts for Tb in mb.Ts]

def align(ma ,mb, pad=5):
    p = pad
    As = alignments(ma,mb)
    h,w = ma.orig.shape
    a = cv.warpPerspective(ma.orig, np.dot(np.eye(3) , desp([p,p])), (w+2*p,h+2*p))
    b = [ (a, cv.warpPerspective(mb.orig, np.dot(desp([p,p]) , t), (w+2*p,h+2*p)) ) for t in As ]
    return b

#########################################################################

def pred(mods,mt,th,similfun):
    win = min([(similfun(m,mt),m) for m in mods], key=lambda x: x[0][0])
    if win[0][0] < th:
        return win[1].lab
    else:
        return '?'

def error_rate(mods,test,th,similfun):
    preds = [(pred(mods, x, th, similfun),x.lab) for x in test]
    rej = len([1 for a,_ in preds if a=='?'])
    acc = len([1 for a,b in preds if a==b])
    n = len(test)
    if False:
        print([a+b for a,b in preds if a!='?' and a!=b])
    return (n-rej-acc)/n, rej/n
    #return (n-rej-acc)/(n-rej), rej/n

def roc(models, targets, similfun, ths):
    e,r = zip(*[error_rate(models,targets, th, similfun) for th in ths])
    return r,e


def selfSimilitude(models,distfun,tot=20):
    return sorted([(distfun(a,b)[0],a.lab+b.lab)
                   for a in models for b in models if a.lab < b.lab ])[:tot]

##########################################################################

def hausdorff(ma,mb, debug=False):
    if debug: #FIXME
        comb = [np.hstack([ma.can[i] * mb.dtc[j], ma.dtc[i] * mb.can[j]]) 
                for i in range(len(ma.can)) for j in range(len(mb.can))]
        win =  min([(c.max(), c) for c in comb ], key=lambda x: x[0])
        return win
    else:
        return min([(ma.auxh1[i] * mb.auxh2[j]).max() for i in range(len(ma.can)) for j in range(len(mb.can))]), None


def hausdorffOrig(ma,mb, debug=False):
    if debug: #FIXME
        comb = [np.hstack([ma.can[i] * mb.dtco[j], ma.dtco[i] * mb.can[j]]) 
                 for i in range(len(ma.can)) for j in range(len(mb.can))]
        win =  min([(c.max(), c) for c in comb ], key=lambda x: x[0])
        return win
    else:
        return min([(ma.auxh3[i] * mb.auxh4[j]).max() for i in range(len(ma.can)) for j in range(len(mb.can))]), None


def xordist(ma,mb):
    comb = [abs(ma.can[i] - mb.can[j]) 
                for i in range(len(ma.can)) for j in range(len(mb.can))]
    win =  min([(c.sum(), c) for c in comb ], key=lambda x: x[0])
    return win


def invardist(ma,mb):
    return la.norm(ma.invar-mb.invar), None


def skfdist(ma,mb):
    return la.norm(ma.skf-mb.skf), None


def rawdist(ma,mb):
    return la.norm(ma.centred - mb.centred), None   # Frobenius


def rawhausdorf(ma,mb):
    return max((ma.dtcen * mb.centred).max(), (mb.dtcen * ma.centred).max()), None


def hausPhys(model,target,pad=5):
    As = align(target,model,pad=pad)
    da = dt(As[0][0])
    thing = [(da * b, a * dt(b)) for a,b in As]
    
    return min(  [ max(a.max(), b.max())  for a,b in thing]), thing


