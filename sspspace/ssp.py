import numpy as np

def sim(a,b):
    return np.einsum('nd,md->nm', a, b)

def bind(a, b):
    '''
    Binds (circular convolution) two sets of vectors together
    '''

    assert a.shape[1] == b.shape[1], f'Expected SSPs to have same dimensionality.  Got {self.shape} * {b.shape}'
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real.view(SSP)
    
def invert(a):
    '''
    Implements the pseudo-inverse of the SSP
    '''
    return a[:,-np.arange(a.ssp_dim)]

def normalize(ssp):
    return ssp.data/np.maximum(np.sqrt(np.sum(ssp**2, axis=1)), 1e-8)

def make_unitary(ssp):
    '''
    Ensures the SSPs are unitary vectors.  See notes for make_unitary_fourier
    '''
    fssp = make_unitary_fourier(np.fft.fft(ssp, axis=1))
    return SSP(np.fft.ifft(fssp, axis=1).real)

def make_unitary_fourier(fssp):
    '''
    Ensures the SSPs are unitary vectors, which ensures that for all phasors, 
    Ae^{i\theta}, the coefficient A=1. This is important to make sure that 
    iterative binding doesn't cause the vector length to change.
    '''
    fssp = fssp/np.maximum(np.sqrt(fssp.real**2 + fssp.imag**2), 1e-8)
    return fssp

def fourier_log(ssp):
    '''
    Computes the log of the Fourier representation 
    '''
    return np.ifft(np.log(np.fft(ssp, axis=1)), axis=1)

class SSP(np.ndarray):
    def __new__(cls, input_array):
        obj = np.atleast_2d(np.asarray(input_array)).view(cls)
        return obj

    def __init__(cls, *args, **kwargs):
        pass

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
        ### TODO: Right now, the scalar has to come after the ssp
        ### need to fix
        if hasattr(other, 'shape'):
            return bind(self, other)
        else:
            return np.multiply(self, other)

    def __or__(self, other):
        return sim(self, other)
    
    def identity(self):
        s = np.zeros(self.shape[1])
        s[0] = 1
        return s

    def unitary(self):
        return make_unitary(self)
