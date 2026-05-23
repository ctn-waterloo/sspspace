import sys
sys.path.append('../')

from sspspace import SSP, RandomSSPSpace
import numpy as np

def test_ssp():
    a  = [[1,2,3,4]]
    a_ssp = SSP(a)

def test_ssp_dim():
    a  = [[1,2,3,4]]
    a_ssp = SSP(a)
    assert a_ssp.ssp_dim == 4

def test_num_pts():
    a  = [[1,2,3,4]]
    a_ssp = SSP(a)
    assert a_ssp.num_pts == 1

def test_addition():
    a  = [[1,2,3,4]]
    a_ssp = SSP(a)
    b_ssp = a_ssp + a_ssp

    assert np.all(b_ssp == 2 * a_ssp)

def test_encoding():
    xs = np.linspace(-10,10,1000).reshape((-1,1))
    rand_encoder = RandomSSPSpace(domain_dim=1,ssp_dim=1024*2)

    query_xs_ssp = rand_encoder.encode(xs)
    origin_ssp = rand_encoder.encode([[0]])

    sims = query_xs_ssp | origin_ssp
    true_sims = np.sinc(xs)
    
    error = np.mean(np.power(sims-true_sims, 2))

    assert error < 0.01, f'Expected error of 0.01, got {error}'

def test_binding():
    xs = np.linspace(-10,10,1000).reshape((-1,1))
    rand_encoder = RandomSSPSpace(domain_dim=1,ssp_dim=1024*2)

    query_xs_ssp = rand_encoder.encode(xs)
    origin_ssp = rand_encoder.encode([[0]])

    bound_self = query_xs_ssp * query_xs_ssp
    sims = bound_self | origin_ssp
    true_sims = np.sinc(2 * xs)
    error = np.mean(np.power(sims-true_sims, 2))

    assert error < 0.01, f'Expected error of 0.01, got {error}'

def test_invert():
    rand_encoder = RandomSSPSpace(domain_dim=1,ssp_dim=1024*2)

    origin_ssp = rand_encoder.encode([[0]])
    origin_inv_ssp = ~origin_ssp

    bound_vec = origin_inv_ssp * origin_ssp

    assert np.isclose(bound_vec[0,0],1), f'Expected 1, got {bound_vec[0,0]}'
    assert np.all(np.isclose(bound_vec[0,1:],0)), f'Expected 1, got {bound_vec[0,1:].max()}'
