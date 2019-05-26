import numpy        as np
import numpy.linalg as la


def einsum(A,B):
    X,x = A
    Y,y = B
    idx = [i for i in x if not i in y] + [i for i in y if not i in x]
    return (np.einsum(x+','+y,X,Y),''.join(sorted(idx)))


def einmul(A,B, contract=''):
    X,x = A
    Y,y = B
    idx = ''.join(sorted([i for i in list(set(x+y)) if not i in contract]))
    return (np.einsum(x+','+y+'->'+idx,X,Y),idx)



IDX='ijklmnopqrstuvwxyzabcdefgh'

def fresh(idx):
    return ''.join([ c for c in IDX if not c in idx ][:len(idx)])

class T():

    def __init__(self,t,idx = None):
        self.A = np.array(t)
        if idx == None:
            self.idx = IDX[:len(self.A.shape)]
        else:
            self.idx = idx

    def __add__(self,other):
        return T(self.A + other.reorder(self.idx).A, self.idx)
    
    def __sub__(self,other):
        return self + T(-other.A,other.idx)
    
    def __matmul__(self,other):
        return T(*einsum((self.A,self.idx),(other.A, other.idx)))

    def __mul__(self,other):
        return T(*einmul((self.A,self.idx),(other.A, other.idx)))

    def __xor__(self,other):
        ren = other
        ren.idx = fresh(ren.idx + self.idx)[:len(ren.idx)]
        return T( asym( (np.dot(self , ren)).A ) )

    def __truediv__(self,other):
        return T(*tensorsolve((other.A,other.idx),(self.A,self.idx)))

    def __call__(self,newidx):
        return T(self.A, newidx)

    def reorder(self,newidx):
        assert(sorted(self.idx)==sorted(newidx))
        return T(np.einsum(self.idx+'->'+newidx,self.A),newidx)

    def __str__(self):
        return self.idx + '\n' + str(self.A)

    def __repr__(self):
        s = 'T('+repr(self.A)+','+repr(self.idx)+')'
        return s.replace('array(','').replace('),',',').replace('\n    ','\n')



def mul(x,y, sum=''):
        return T(*einmul((x.A,x.idx),(y.A, y.idx),sum))



def unzip(l):
    if l:
        return list(zip(*l))
    else:
        return (),()

def tensorsolve(A,B):
    X,x = A
    Y,y = B
    ixy, cxy = unzip([ (k,a) for k,a in enumerate(x) if a in y ])
    ix,  cx  = unzip([ (k,a) for k,a in enumerate(x) if a not in y ])
    iyx, cyx = unzip([ (k,a) for k,a in enumerate(y) if a in x ])
    iy,  cy  = unzip([ (k,a) for k,a in enumerate(y) if a not in x ])
    xr = np.array(X.shape)[list(ixy)]
    xc = np.array(X.shape)[list(ix)]
    yr = np.array(Y.shape)[list(iyx)]
    yc = np.array(Y.shape)[list(iy)]
    XM = X.transpose(list(ixy)+list(ix)).reshape(np.product(xr),np.product(xc))
    YM = Y.transpose(list(iyx)+list(iy)).reshape(np.product(yr),np.product(yc))
    S  = la.lstsq(XM,YM)
    print("\x1b[31m",xr,xc,yr,yc,XM.shape,YM.shape,"\x1b[0m")
    return S[0].reshape(list(xc)+list(yc)), ''.join(cx+cy)

def null1(M):
    u,s,vt = la.svd(M)
    return vt[-1,:]

def nullTensor(A,si):
    X,x = A.A, A.idx
    ix,  cx  = unzip([ (k,a) for k,a in enumerate(x) if a not in si ])
    iy,  cy  = unzip([ (k,a) for k,a in enumerate(x) if a     in si ])
    xr = np.array(X.shape)[list(ix)]
    xc = np.array(X.shape)[list(iy)]
    XM = X.transpose(list(ix)+list(iy)).reshape(np.product(xr),np.product(xc))
    print("\x1b[31m",xr,xc,XM.shape,la.matrix_rank(XM),"\x1b[0m")
    S  = null1(XM)
    return T(S.reshape(list(xc)), ''.join(cy))

#############################################################################

from itertools import permutations

def sym(a):
    return sum(a.transpose(p) for p in permutations(range(len(a.shape))))


def perm_parity(lst):
    # http://code.activestate.com/recipes/578227-generate-the-parity-or-sign-of-a-permutation/

    '''\
    Given a permutation of the digits 0..N in order as a list, 
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity


def asym(a):
    return sum(a.transpose(p) * perm_parity(list(p)) for p in permutations(range(len(a.shape))))


def prod(l):
    if not l: return T(1)
    r = l[0]
    for t in l[1:]:
        r = r.__matmul__(t)
    return r

def LeviCivita(n):
    basis = [T(v,i) for v,i in zip( list(np.eye(n)), IDX ) ]
    return T(asym(prod(basis).A))

eps3 = LeviCivita(3)
eps4 = LeviCivita(4)


def parts(t,i):
    ri = ''.join([p for p in t.idx if p != i])
    rt = t.reorder(i+ri)
    return [T(x,ri) for x in list(rt.A)]
    
def join(l,i):
    assert l
    idx = l[0].idx
    return T(np.array([ t.reorder(idx).A for t in l ]), i+idx)

def mapAt(f,t,i):
    return join([f(x) for x in parts(t,i)],i)

