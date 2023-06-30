import matplotlib.pyplot as plt
import numpy as np
import sspspace
import pytest
import sys,os

from sklearn.metrics import mean_absolute_percentage_error

@pytest.mark.parametrize( 'method' , ['nef-network'] )
@pytest.mark.parametrize( 'domain_dim_' , [1,2,3] )
@pytest.mark.parametrize( 'n_test_points' , [1,100,1000] )
def test_decoder(method,n_test_points,domain_dim_, plot=False):

    # SSP encoding parameters
    ssp_ls_ = 1.
    ssp_dim_ = 2048

    # instantiate the ssp encoder object
    domain_bounds_ = np.tile( [-10,10], (domain_dim_,1) ).T

    encoder = sspspace.RandomSSPSpace(
                                domain_dim = domain_dim_,
                                ssp_dim = ssp_dim_,
                                )
    encoder.update_lengthscale(ssp_ls_)

    test_xs = np.random.uniform( domain_bounds_[0,:],domain_bounds_[1,:],size = (n_test_points,domain_dim_) )
    data_phis = encoder.encode(test_xs)
    
    # decoding method #1: query similarity
    if method == 'similarity':

        # generate set of 2d query points
        n_query_points = 100
        meshes = np.meshgrid(*[np.linspace(b[0], b[1], n_query_points) 
                                for b in domain_bounds_.T])
        query_xs = np.vstack([m.flatten() for m in meshes]).T
        
        # project query points into HexSSP space
        query_phis = encoder.encode(query_xs)
        
        decoder = sspspace.decoders.SSPSimilarityDecoder(sim_xs = query_xs,sim_ssps = query_phis,encoder = encoder)
        
        decoded_xs = decoder.decode(data_phis)
        assert decoded_xs.shape == test_xs.shape

    # decoding method #2: train a deep network
    elif method == 'sk-network':

        print('training decoder network ...')
        decoder, training_history = sspspace.decoders.train_decoder_net_sk(encoder, 
                                        bounds = domain_bounds_.T, 
                                        n_training_pts = 200000, # number of points sample from domain
                                        hidden_units = [8], # list of neurons in dense hidden layers
                                        learning_rate = 1e-3, # learning rate for Adam
                                        n_epochs = 50, # Max epochs
                                        patience = 3, # patience for early stopping
                                        verbose = True # print tensorflow training nonsense.
                                    )

        decoded_xs = decoder.decode(data_phis)
        assert decoded_xs.shape == test_xs.shape

    # decoding method #3: optimize a shallow network to perform the mapping
    elif method == 'nef-network':
        
        decoder,_ = sspspace.decoders.train_decoder_nef(encoder,bounds = domain_bounds_)
    
        decoded_xs = decoder.decode(data_phis, optimize = True)
        assert decoded_xs.shape == test_xs.shape

    if plot == True:
        fig,axs = plt.subplots(1,domain_dim_,squeeze=False,figsize=(3*domain_dim_,3))
        for d,ax in enumerate(axs.ravel()):
            ax.scatter(test_xs[:,d],decoded_xs[:,d],color='k',alpha=.5)
            
            b0 = domain_bounds_[0,d]
            b1 = domain_bounds_[1,d]
            ax.plot( [b0,b1],[b0,b1], color='dimgray',linestyle='--')
            ax.set_xlim( b0,b1 )
            ax.set_xlabel('True, dim {}'.format(d))
            ax.set_ylim( b0,b1 )
            ax.set_ylabel('Decoded, dim {}'.format(d))
        fig.tight_layout()
        plt.show()

    error = mean_absolute_percentage_error(test_xs,decoded_xs)   
    print('error: ', error)
    assert error < 0.1
        
if __name__ == '__main__':

    method = sys.argv[1]
    n_test_points = int(sys.argv[2])
    domain_dim_ = int(sys.argv[3])
    
    test_decoder(method = method, n_test_points = n_test_points, domain_dim_ = domain_dim_, plot = True )