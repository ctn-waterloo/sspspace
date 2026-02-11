import numpy as np

def sim(a : np.ndarray, b : np.ndarray):
    return np.einsum('nd,md->nm', a, b)

def bind(a : np.ndarray, b : np.ndarray):
    '''
    Binds (circular convolution) two sets of vectors together
    '''

    assert a.shape[1] == b.shape[1], f'Expected SSPs to have same dimensionality.  Got {a.shape} * {b.shape}'
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real.view(SSP)
    
def invert(a : np.ndarray):
    '''
    Implements the pseudo-inverse of the SSP
    '''
    return a[:,-np.arange(a.ssp_dim)]

def normalize(ssp):
    return SSP(ssp.v/np.maximum(np.sqrt(np.sum(ssp**2, axis=1)), 1e-8))

def make_unitary(ssp):
    '''
    Ensures the SSPs are unitary vectors.  See notes for make_unitary_fourier
    '''
    fssp = make_unitary_fourier(np.fft.fft(ssp.v, axis=1))
    return SSP(np.fft.ifft(fssp, axis=1).real)

def make_unitary_fourier(fssp : np.ndarray):
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
    return np.ifft(np.log(np.fft(ssp.v, axis=1)), axis=1)

class SSP:
    def __init__(self, input_array):
        self.v = np.copy(input_array)

    @property
    def ssp_dim(self):
        return self.v.shape[1]
       
    @property
    def num_pts(self):
        return self.v.shape[0]

    def __invert__(self):
        return SSP(invert(self.v))

    def __add__(self, other):
        assert self.v.shape == other.v.shape, f'Expected arguments to have the same shape but got {self.v.shape} and {other.v.shape}'
        return SSP(self.v + other.v)

    def __mul__(self, other):
        if hasattr(other, 'v'):
            return SSP(bind(self.v, other.v))
        else:
            return SSP(np.multiply(self.v, other))

    def __or__(self, other):
        return sim(self.v, other.v)
    
    def identity(self):
        s = np.zeros(self.shape[1])
        s[0] = 1
        return s

    def unitary(self):
        return make_unitary(self.v)
