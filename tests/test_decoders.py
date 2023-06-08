import matplotlib.pyplot as plt
import numpy as np
import sspspace
import pytest
import sys,os

from sklearn.metrics import mean_squared_error

@pytest.mark.parametrize( 'method' , ['similarity','network'] )
@pytest.mark.parametrize( 'n_test_points' , [1,100,1000] )
def test_decoder_2d(method,n_test_points):

    # domain parameters
    domain_dim_ = 2

    # generate set of 2d query points
    n_query_points = 100
    query_dim0 = np.linspace(-10., 10., n_query_points)
    query_dim1 = np.linspace(-10., 10., n_query_points)
    D0,D1 = np.meshgrid(np.sort(query_dim0), np.sort(query_dim1))
    query_xs = np.dstack((D0, D1)).reshape((-1,2))

    # SSP encoding parameters
    # length_scale_ = 0.1   does a random encoder not have a length scale?
    ssp_dim_ = 1024

    # instantiate the ssp encoder object
    domain_bounds_ = np.tile([-10.,10.],(domain_dim_,1))
    rand_encoder = sspspace.RandomSSPSpace(
                                domain_dim = domain_dim_,
                                ssp_dim = ssp_dim_,
                                )

    print('testing decoders by uniformly sampling {} points from the domain ... '.format(n_test_points))
    test_xs = np.random.uniform( domain_bounds_[:,0],domain_bounds_[:,1],size = (n_test_points,domain_dim_) )
    data_phis = rand_encoder.encode(test_xs)
    
    # decoding method #1: query similarity
    if method == 'similarity':
        # project query points into HexSSP space
        print('encoding query points for the similarity decoder ...')
        query_phis = rand_encoder.encode(query_xs)
        
        sims = np.einsum('nd,md->nm', query_phis, data_phis)
        decoded_xs = query_xs[np.argmax(sims,axis=0),:]
        assert decoded_xs.shape == test_xs.shape
        
        error = mean_squared_error(test_xs, decoded_xs)
        assert error < .01

    # decoding method #2: use the trained decoder
    elif method == 'network':

        print('training decoder network ...')
        rand_decoder, training_history = sspspace.train_decoder_net_sk(rand_encoder, 
                                        bounds = domain_bounds_, 
                                        n_training_pts = 200000, # number of points sample from domain
                                        hidden_units = [8], # list of neurons in dense hidden layers
                                        learning_rate = 1e-3, # learning rate for Adam
                                        n_epochs = 20, # Max epochs
                                        patience = 3, # patience for early stopping
                                        verbose = True # print tensorflow training nonsense.
                                    )

        decoded_xs = rand_decoder.decode(data_phis)
        assert decoded_xs.shape == test_xs.shape
        
        error = mean_squared_error(test_xs, decoded_xs) 
        print('error:', error)
        assert error < .01
        
if __name__ == '__main__':

    method = sys.argv[1]
    n_test_points = int(sys.argv[2])
    
    test_decoder_2d( method = method, n_test_points = n_test_points )