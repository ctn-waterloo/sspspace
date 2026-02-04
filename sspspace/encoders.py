from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union

import numpy as np
from scipy.stats import semicircular, chi
from .ssp import SSP
from .util import make_good_unitary, conjugate_symmetry, vecs_from_phases

def k_to_vector(K):
    fs = vecs_from_phases(K.T).T
    phis = np.fft.ifft(fs, axis=1).real
    return phis


class DiscreteSPSpace:
    def __init__(self, keys, ssp_dim):
        self.ssp_dim = ssp_dim 
        self.length_scale = np.array([1])
        self.keys = keys
#         self.map = SSP([make_good_unitary(ssp_dim) for k in self.keys])

        self.map = SSP(np.zeros((len(self.keys), self.ssp_dim)))

        phase0 = np.random.uniform(low=-np.pi, high=np.pi, 
                                   size=(1, (self.ssp_dim-2)//2))

        self.map[0,:] = k_to_vector(phase0)
        
        def greedy_min_func(x, vecs):
            K = x.reshape((1,vecs.shape[1]//2 - 1))
            phi = k_to_vector(K)    
            sims = np.einsum('nd,md->nm', phi, vecs)
            return np.linalg.norm(sims)

        from scipy.optimize import minimize
       
        for i in range(1,len(self.keys)):
            x0 = np.random.uniform(low=-np.pi, 
                                   high=np.pi, 
                                   size=((self.ssp_dim -2)// 2,))
            greedy_soln = minimize(greedy_min_func, x0, 
                                   args=(self.map[:i,:]), 
                                   method='L-BFGS-B')
            self.map[i,:] = k_to_vector(greedy_soln.x.reshape((1,(self.ssp_dim-2)//2)))
    ### end __init__


    def encode(self, vals):
        retval = np.zeros((vals.shape[0], self.ssp_dim))
        for v_idx, v in enumerate(vals): 
            if v not in self.keys:
                raise RuntimeWarning(f'Key {v} is not in the dictionary')
            retval[v_idx,:] = self.map[self.keys.index(v),:].reshape((1,-1))
        return SSP(retval)

    def decode(self, ssp):
        return self.keys[np.argmax(self.map | ssp)]

        
class SSPEncoder:
    def __init__(self, phase_matrix:np.ndarray, length_scale:Optional[Union[int, np.ndarray]]=1):
        '''
        Represents a domain using spatial semantic pointers.

        Parameters:
        -----------

        phase_matrix : np.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.

        length_scale : float or np.ndarray
            Scales values before encoding.
            
        '''
        self.phase_matrix = phase_matrix
        self.domain_dim = self.phase_matrix.shape[1]
        self.ssp_dim = self.phase_matrix.shape[0]
        self.length_scale = length_scale * np.ones((self.domain_dim,1))
        
        self.phase_matrix = phase_matrix

    def update_lengthscale(self, scale):
        '''
        Changes the lengthscale being used in the encoding.
        '''
        if not isinstance(scale, np.ndarray) or scale.size == 1:
            self.length_scale = scale * np.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim
        ### end if
        
    def encode(self,x):
        '''
        Transforms input data into an SSP representation.

        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the data
            
        '''

        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        # TODO: add conditional debugging catch for non-zero imaginary components of the data.
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T), axis=0 ).real
        return SSP(data.T)

    def gradient(self, phi):
        '''
        Returns the gradient of an encoded SSP.  

        Parameters:
        -----------

        phi : SSP
            An SSP object representing a single SSP. i.e., has shape (1, ssp_dim)

        Returns:
        --------

        grad : np.array

            A (domain_dim, ssp_dim) np.array that represents the gradient of the encoding at the encoded value.

        '''

        phi_fourier = np.fft.fft(phi, axis=1)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale.flatten()))
        # d/dx[e^iAx] = hadamard(iA, e^{iAx})
        deriv_mat = 1.j * (self.phase_matrix @ ls_mat) # Derivative coeff

        fourier_grad = np.einsum('dm,d->md',deriv_mat,phi_fourier.flatten())
        return np.fft.ifft(fourier_grad, axis=1).real

    
    def encode_and_deriv(self,x):
        '''
        Returns the ssp representation of the data and the derivative of
        the encoding.

        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the 
            data

        grad : np.ndarray
            A (num_samples, ssp_dim, domain_dim) array of the ssp representation of the data

        '''
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale))
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T ), axis=0 ).real
        ddata = np.fft.ifft( 1.j * (self.phase_matrix @ ls_mat) @ np.exp( 1.j * self.phase_matrix @ scaled_x.T ), axis=0 ).real
        return SSP(data.T), ddata.T
    
    def encode_fourier(self,x):
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.exp( 1.j * self.phase_matrix @ scaled_x.T)

        return data.T
    
    
    
# Make Encoder Matrices
def RandomSSPSpace(domain_dim:int, ssp_dim:int, 
                   length_scale:Optional[Union[int, np.ndarray]]=1, 
                   rng=np.random.default_rng(), kernel="sinc"):
    assert kernel in ["sinc", "gaussian", "jinc"], f"Kernel \"{kernel}\" is not in supported"
    
    phase_matrix=np.zeros((ssp_dim, domain_dim))
    if kernel == "sinc":
        phase_samples = rng.uniform(-1, 1, size=((ssp_dim - 1)//2, domain_dim))
    elif kernel == "gaussian":
        phase_samples = rng.normal(0, 1, size=((ssp_dim - 1)//2, domain_dim))
    else:
        phase_samples = semicircular.rvs(0, 1, size=((ssp_dim - 1)//2, domain_dim))
    
    phase_matrix[1:(ssp_dim + 1) // 2,:] = phase_samples
    phase_matrix[-1:ssp_dim // 2:-1] = -phase_matrix[1:(ssp_dim + 1) // 2,:]
    
    return SSPEncoder(phase_matrix, length_scale=length_scale)

def CyclicSSPSpace(domain_dim:int, ssp_dim:int, period:float,
                   band_scale:Optional[Union[int, np.ndarray]]=1,
                   rng=np.random.default_rng(), kernel="sinc"):
    assert kernel in ["sinc", "gaussian", "jinc"], f"Kernel \"{kernel}\" is not in supported"
    
    phase_matrix = np.zeros((ssp_dim, domain_dim))
    if kernel == "sinc":
        scales = rng.uniform(-1, 1, size=((ssp_dim - 1) // 2, domain_dim))
    elif kernel == "gaussian":
        scales = rng.normal(0, 1, size=((ssp_dim - 1) // 2, domain_dim))
    else:
        scales = semicircular.rvs(0, 1, size=((ssp_dim - 1) // 2, domain_dim))
    
    scales = (period / band_scale) * scales
    int_scales = ((2 * np.pi) / period) * np.floor(scales)

    phase_matrix[1:(ssp_dim + 1) // 2,:] = int_scales
    phase_matrix[-1:ssp_dim // 2:-1] = -phase_matrix[1:(ssp_dim + 1) // 2,:]

    return SSPEncoder(phase_matrix, length_scale=1)

def HexagonalSSPSpace(domain_dim:int, 
                      n_rotates:int=5, 
                      n_scales:int=5, 
                      kernel="jinc",
                      length_scale:Optional[Union[int, np.ndarray]]=1):
    '''
    Creates an SSP space using the Hexagonal Tiling developed by NS Dumont 
    (2020)
    '''
    assert kernel in ["sinc", "gaussian", "jinc"], f"Kernel \"{kernel}\" is not in supported"
    phases_hex = np.hstack([np.sqrt(1+ 1/domain_dim)*np.identity(domain_dim) - (domain_dim**(-3/2))*(np.sqrt(domain_dim+1) + 1),
                         (domain_dim**(-1/2))*np.ones((domain_dim,1))]).T

    scales = np.linspace(0, 1, (n_scales if domain_dim != 1 else n_scales + n_rotates) + 1, endpoint=False)[1:]
    
    if kernel == "sinc":
        scales = scales ** (1/domain_dim)
    elif kernel == "gaussian":
        scales = chi.ppf(scales, df=domain_dim, loc=0, scale=1)
    else:
        pass 
    phases_scaled = np.vstack([phases_hex*i for i in scales])

    if (n_rotates == 1 or domain_dim == 1):
        phases_scaled_rotated = phases_scaled
    elif (domain_dim == 2):
        angles = np.linspace(0,2*np.pi/3,n_rotates,endpoint=False)
        R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                        np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
        phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
    else:
        R_mats = special_ortho_group.rvs(domain_dim, size=n_rotates, random_state=1)
        phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        
    phase_matrix = conjugate_symmetry(phases_scaled_rotated)

    return SSPEncoder(phase_matrix, length_scale=length_scale)

