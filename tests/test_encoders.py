import sys
sys.path.append('../')

from sspspace import SSP, SSPEncoder, RandomSSPSpace, HexagonalSSPSpace
import numpy as np


def test_random_encoding():
    xs = np.linspace(-10,10,1000).reshape((-1,1))
    rand_encoder = RandomSSPSpace(domain_dim=1,ssp_dim=1024*2)

    query_xs_ssp = rand_encoder.encode(xs)
    origin_ssp = rand_encoder.encode([[0]])

    sims = query_xs_ssp | origin_ssp
    true_sims = np.sinc(xs)
    
    error = np.mean(np.power(sims-true_sims, 2))

    assert error < 0.01, f'Expected error of 0.01, got {error}'

def test_hexagonal_encoding():
    xs = np.linspace(-2,2,100).reshape((-1,1))
    ys = np.linspace(-2,2,100).reshape((-1,1))

    X, Y = np.meshgrid(xs, ys)
    xys = np.vstack((X.flatten(), Y.flatten())).T
    hex_encoder = HexagonalSSPSpace(domain_dim=2, n_rotates=1, n_scales=1)
    hex_encoder.update_lengthscale(0.01)

    query_xys_ssp = hex_encoder.encode(xys)
    origin_ssp = hex_encoder.encode([[0,0]])

    sims = query_xys_ssp | origin_ssp
    square_sims = sims.reshape((ys.shape[0], xs.shape[0]))

    # TODO: Figure out numerical comparison

def test_gradient():

    x = np.array([[1,1]])
    rand_encoder = RandomSSPSpace(domain_dim=2,ssp_dim=128)

    phi = rand_encoder.encode(x)
    grad = rand_encoder.gradient(phi)
    print(grad)

#     import matplotlib.pyplot as plt
#     plt.imshow(square_sims)
#     plt.show()


# def test_binding():
#     xs = np.linspace(-10,10,1000).reshape((-1,1))
#     rand_encoder = RandomSSPSpace(domain_dim=1,ssp_dim=1024*2)
# 
#     query_xs_ssp = rand_encoder.encode(xs)
#     origin_ssp = rand_encoder.encode([[0]])
# 
#     bound_self = query_xs_ssp * query_xs_ssp
#     sims = bound_self | origin_ssp
#     true_sims = np.sinc(2 * xs)
#     error = np.mean(np.power(sims-true_sims, 2))
# 
#     assert error < 0.01, f'Expected error of 0.01, got {error}'
# 
# def test_invert():
#     rand_encoder = RandomSSPSpace(domain_dim=1,ssp_dim=1024*2)
# 
#     origin_ssp = rand_encoder.encode([[0]])
#     origin_inv_ssp = ~origin_ssp
# 
#     bound_vec = origin_inv_ssp * origin_ssp
# 
#     assert np.isclose(bound_vec[0,0],1), f'Expected 1, got {bound_vec[0,0]}'
#     assert np.all(np.isclose(bound_vec[0,1:],0)), f'Expected 1, got {bound_vec[0,1:].max()}'
