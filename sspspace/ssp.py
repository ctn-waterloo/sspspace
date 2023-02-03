import numpy as np

def bind(a, b):
    assert a.shape == b.shape, f'Expected operators have same dim.  Got {self.shape} * {b.shape}'
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real.view(SSP)
    
def invert(a):
    return a[:,-np.arange(a.ssp_dim)]

def normalize(ssp):
    return ssp.data/np.maximum(np.sqrt(np.sum(ssp**2, axis=1)), 1e-8)

def make_unitary(ssp):
    fssp = make_unitary_fourier(np.fft.fft(ssp, axis=1))
    return SSP(np.fft.ifft(fssp, axis=1).real)

def make_unitary_fourier(fssp):
    fssp = fssp/np.maximum(np.sqrt(fssp.real**2 + fssp.imag**2), 1e-8, axis=1)
    return fssp

class SSP(np.ndarray):
    def __new__(cls, input_array):
        obj = np.atleast_2d(np.asarray(input_array)).view(cls)
        return obj

    def __init__(cls, *args, **kwargs):
        print(f'In __init__ with class {cls.__class__}')

    def __array_finalize__(self, obj):
        if obj is None: return

    @property
    def ssp_dim(self):
        return self.shape[1]
       
    @property
    def num_pts(self):
        return self.shape[0]

    def __invert__(self):
        return invert(self)

    def __mul__(self, other):
        return bind(self, other)

    def __or__(self, other):
        return np.einsum('nd,md->nm', self, other)
    
    def identity(self):
        s = np.zeros(self.shape[1])
        s[0] = 1
        return s
