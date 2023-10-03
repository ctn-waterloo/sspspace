import numpy as np

from scipy.stats import qmc

def sample_domain(bounds, num_samples, scramble=True):
    sampler = qmc.Sobol(d=bounds.shape[0], scramble=scramble) 
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    m = int(np.log(num_samples) / np.log(2))
    u_sample_points = sampler.random_base2(m)
    sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
    return sample_points

def make_good_unitary(dim, eps=1e-3, rng=np.random):
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v

def conjugate_symmetry(K):
    d = K.shape[0]
    F = np.ones((d*2 + 1,K.shape[1]), dtype="complex")
    F[0:d,:] = np.exp(1.j*K)
    F[-d:,:] = np.flip(np.conj(F[0:d,:]),axis=0)
    return F

def vecs_from_phases(K):
    d = K.shape[0]
    F = np.ones((d*2+2, K.shape[1]), dtype="complex")
    F[1:d+1,:] = np.exp(1.j*K)
    F[-d:,:] = np.flip(np.conj(F[1:d+1,:]),axis=0)
    F[0,:] = 1
    F[d+1,:] = 1
    return F


def similarity_plot(self,ssp,n_grid=100,plot_type='heatmap',ax=None,**kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    if self.domain_dim == 1:
        xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
        sims = ssp @ self.encode(np.atleast_2d(xs).T).T
        im = ax.plot(xs, sims.reshape(-1) )
        ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
    elif self.domain_dim == 2:
        xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
        ys = np.linspace(self.domain_bounds[1,0],self.domain_bounds[1,1], n_grid)
        X,Y = np.meshgrid(xs,ys)
        sims = ssp @ self.encode(np.vstack([X.reshape(-1),Y.reshape(-1)]).T).T 
        if plot_type=='heatmap':
            im = ax.pcolormesh(X,Y,sims.reshape(X.shape),**kwargs)
        elif plot_type=='contour':
            im = ax.contour(X,Y,sims.reshape(X.shape),**kwargs)
        elif plot_type=='contourf':
            im = ax.contourf(X,Y,sims.reshape(X.shape),**kwargs)
        ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
        ax.set_ylim(self.domain_bounds[1,0],self.domain_bounds[1,1])
    else:
        raise NotImplementedError()
    return im
