import numpy        as np
import numpy.linalg as la
import cv2          as cv

import umucv.htrans as ht
from umucv.htrans import htrans, vec,row,col,rot3,jc,desp,scale,depthOfPoint, rotation, Pose
from umucv.contours import orientation, fixOrientation, extractContours
from umucv.contours import autoscale, whitener, fourierPL
from umucv.util import ejeX, ejeY, ejeZ, cube
from numpy.fft import fft
from scipy.interpolate import interp1d

def spectralFeat(x, maxw=10):
    f = fourierPL(x)
    return np.array( [ f(w) for w in range(-maxw, maxw+1) ] )


def spectralFeat(x,wmax):
    n = len(x)
    z = x[:,0] + x[:,1]*1j
    Z = z[np.arange(-1,n+1)%n]
    
    dz = Z[1:] - Z[:-1]
    
    dta = abs(dz)
    dta[0] = 0
    t = np.cumsum(dta)
    tot = t[-1]
    t = t / tot
    N = 128
    nt = np.arange(N)/(N)
    F = fft(interp1d(x=t, y=Z[1:], assume_sorted=True, axis=0)(nt))/N
    return F[np.arange(-wmax,wmax+1)%N]



def spectralFeat(x,wmax):
    n = len(x)
    z = x[:,0] + x[:,1]*1j
    Z = z[np.arange(n+2)%n]
    
    dz = Z[1:] - Z[:-1]
    
    dta = abs(dz)
    t = np.cumsum(dta)
    tot = t[-2]
    t = t / tot
    dt = dta / tot
    
    mz =  Z[1:] + Z[:-1]
    f0 = np.sum (mz[:-1]*dt[:-1]) / 2
    
    alpha = dz/dt
    
    A = alpha[:-1] - alpha[1:]
    
    H = np.exp(-2*np.pi*1j*t[:-1])
    w = np.arange(-wmax,wmax+1).reshape(-1,1)
    w[wmax] = 1
    HW = H.reshape(1,-1) ** w / (2*np.pi * w)**2
    r = HW @ A
    r[wmax] = f0
    return r



def invar(v):
    n = (len(v)-1) // 2
    u = abs(v)
    s = u[n-1]+u[n+1]
    return np.delete(u/s,[n-1,n,n+1])

def normalizeStart(v):
    n = (len(v)-1) // 2
    k = np.arange(-n,n+1)
    t = np.angle(v[n+1]-np.conj(v[n-1]))
    vn = np.exp(-1j * t * k) * v
    return vn

def rotfeat(a,v):
    return normalizeStart(np.exp(1j * a) * v)


def vecfeat(f):
    return np.hstack([f.real,f.imag])

def afeat(x,wmax):
    return vecfeat(normalizeStart(spectralFeat(x,wmax)))


# full projective
def mktP(v):
    a,d,c,b,e,f,g,h = v
    return np.array ([ [ 1+a, c,   e],
                       [ b  , 1+d, f],
                       [g  ,   h,  1]])    

## GN

def numjaco(f,z,h):
    def d(k):
        d = np.zeros(len(z))
        d[k] = h
        return d
    fz = f(z)
    ds = [d(k) for k in range(len(z))]
    return fz, np.vstack([ (f(z+d)-fz)/h for d in ds ] ).transpose()


def ICModel(feat, nin):
    z = np.zeros([nin])
    f0,J0 = numjaco(feat,z,1e-4)
    iJ0 = np.dot(la.pinv(np.dot(J0.transpose(),J0)) , J0.transpose())
    return f0,J0,iJ0


def GNModelP(x, wmax):
    f = lambda v: afeat(htrans(mktP(v),x),wmax)
    return ICModel(f,8)



def mkRotator(angs, wmax):
    freqs = np.array(range(-wmax,wmax+1))
    angs = [ k * 2*np.pi/angs for k in range(angs) ]
    return np.array([ [np.exp(1j*(a - w*a)) for a in angs] for w in freqs ]) 


def mkMatcher(angs, wmax):
    rotator = mkRotator(angs,wmax)

    class model:
        def __init__(self, contour):
            self.original = contour
            self.w        = whitener(contour)
            self.white    = htrans(self.w,contour)
            self.z0       = col(normalizeStart(spectralFeat(self.white,wmax)))
            self.gno      = GNModelP(contour,wmax)
            self.gnw      = GNModelP(self.white,wmax)

    class target:
        def __init__(self, contour):
            self.original = contour
            self.w        = whitener(contour)
            self.white    = htrans(self.w,contour)
            f0            = normalizeStart(spectralFeat(self.white,wmax))
            self.z0s      = col(f0) * rotator

    def match(m,t,steps=1):
        errs = t.z0s - m.z0
        aes  = np.sum(abs(errs),0)
        k    = np.argmin(aes)
        err  = errs[:,k]
        R    = rot3(k*2*np.pi/angs)
        aesn = aes[k] / la.norm(m.z0)
        #if aesn > 0.4: return 1, ()

        f0,j0,ij = m.gnw
        dv = np.dot(ij , vecfeat(err))
        dt = mktP(dv)
        
        h = np.dot(la.inv(np.dot(dt , m.w)) , np.dot(R , t.w))
        err = aesn
        
        f0,j0,ij = m.gno
        for q in range(steps):
            s   = htrans(h, t.original)
            err = afeat(s,wmax)-f0
            dv  = np.dot(ij , err)
            dt  = mktP(dv)
            h   = np.dot(la.inv(dt) , h)
            err = la.norm(err)/la.norm(f0)

        return err, h

    return model, target, match
        
##############################################################################

def readrgb(filename):
    return cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

import glob
import sys

def biggest(xs):
    return sorted(xs, key=lambda x: - abs(orientation(x)))[0]

marker = np.array(
       [[0,   0   ],
        [0,   1   ],
        [0.5, 1   ],
        [0.5, 0.5 ],
        [1,   0.5 ],
        [1,   0   ]])

def readModels(path):
    fmods = sorted([name for name in glob.glob(path+'/*.*') if name[-3:] != 'txt'])
    models = sum([ [biggest(extractContours(rgb2gray(readrgb(f))))] for f in fmods ],[])
    if not models:
        models = [marker*[1,-1]]
    return [fixOrientation(autoscale(c.astype(float))) for c in models]


##############################################################################


def move(T, ref):
    uc = np.zeros(ref.shape[:-1]+(1,))
    ref3D = np.append(ref,uc,axis=-1)
    return htrans(T,ref3D)


def getCam(K,h,c,p3d):
    view = htrans(h,c)
    P = Pose(K,view,p3d)
    return P.rms, P.M, P.R, P.C, P.view


class Position:
    def __init__(self, models, angs=30, 
                               wmax=10,
                               verbose=False,
                               ):
        model, self.target, self.match = mkMatcher(angs, wmax)
        self.mod = [model(m) for m in readModels(models)]
        for k,m in enumerate(self.mod):
            m.id = k

        try:
            locs = np.loadtxt(models+'/locations.txt').reshape(-1,7)
            if verbose:
                print(locs)
            T = []
            for ax,ay,az, dx,dy,dz, s in locs:
                t = rotation(vec(1,0,0),np.radians(ax), homog=True)
                t = rotation(vec(0,1,0),np.radians(ay), homog=True) @ t
                t = rotation(vec(0,0,1),np.radians(az), homog=True) @ t
                t = scale(vec(s,s,s)) @ t
                t = desp(vec(dx,dy,dz)) @ t
                T.append(t)
        except:
            T = [ np.eye(4) for _ in self.mod ]

        self.p3d = [ move(t, (m.original*[[1,-1]])) for t,m in zip(T, self.mod) ]
        if verbose:
            print(self.p3d)

    def compute(self, cs, K, errA=0.04, errP=2, steps=2):
        cans = [self.target(c) for c in cs]
        raw = [ (self.match(m,c,steps),(m,c)) for m in self.mod for c in cans ]
        pre = [ (t,mc) for (e0, t), mc in raw if e0 < errA ]
        canposes = [ (getCam(K,la.inv(t),m.original, self.p3d[m.id]), m.id) for (t, (m,c)) in pre ]
        poses = [ (p, R, cen.flatten() , vw, ident) for (err,p,R,cen,vw), ident in canposes if err < errP]
        poses = sorted(poses,key=lambda p: -abs(orientation(p[3])))
        return poses


    def consolidate(self,loc,K):
        assert loc, "can't consolidate a null list of positions"
        ids = [x[-1] for x in loc]
        sviews = np.vstack( [l[3] for l in loc ] )
        sworld = np.vstack( [self.p3d[l[4]] for l in loc ])
        return Pose(K, sviews, sworld)
 

