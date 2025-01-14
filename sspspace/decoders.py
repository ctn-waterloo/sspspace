import numpy as np
from .util import sample_domain

from scipy.optimize import minimize

class SSPDecoder:
    def __init__(self, domain_bounds, model, encoder):
        self.decoder_network = model 
        self.encoder = encoder
        self.domain_bounds = domain_bounds
        pass

    def decode(self, ssps, optimize = True):

        x0 = self.decoder_network.predict(ssps).reshape((-1,self.encoder.domain_dim))
        
        if optimize == True:
            solns = np.zeros(x0.shape)
            for i in range(x0.shape[0]):
                def min_func(x,target=ssps[i,:]):
                    x_ssp = self.encoder.encode(np.atleast_2d(x))
                    return -np.inner(x_ssp, target).flatten()
                soln = minimize(min_func, x0[i,:], 
                            method='L-BFGS-B',
    #                         bounds=self.domain_bounds,
                )
                solns[i,:] = soln.x
            return solns
        
        else:
            return x0
    ### end decode

class SSPSimilarityDecoder: 

    def __init__(self, sim_xs, sim_ssps, encoder, optim=True):
        self.sim_xs = sim_xs
        self.sim_ssps = sim_ssps
        self.encoder = encoder
        self.optim = optim

    def decode(self, ssps, optimize = True):
        sims = self.sim_ssps | ssps
        
        x0 = self.sim_xs[np.argmax(sims, axis=0),:]
        if self.optim == True and optimize == True:
            solns = np.zeros(x0.shape)
            for i in range(x0.shape[0]):
                def min_func(x,target=ssps[i,:]):
                    x_ssp = self.encoder.encode(np.atleast_2d(x))
                    return -np.inner(x_ssp, target).flatten()
                soln = minimize(min_func, x0[i,:], 
                            method='L-BFGS-B',
    #                         bounds=self.domain_bounds,
                )
                solns[i,:] = soln.x
            return solns
        else:
            return x0

def train_decoder_net_sk(encoder, bounds, n_training_pts = 200000, 
                        corrupt_training_examples = True,
                        gaussian_noise_std = 'auto',
                        sample_points = None,
                        hidden_units = [8],
                        learning_rate = 1e-3, 
                        n_epochs = 100, 
                        patience = 3,
                        tolerance = 1e-6,
                        verbose = True):
    '''
        Trains a dense neural network to decode SSPs.

    '''
    import sklearn
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(hidden_layer_sizes = hidden_units,
                         activation='relu',
                         solver='adam',
                         alpha=0.0001,
                         batch_size='auto',
                         learning_rate_init=learning_rate,
                         tol=tolerance,
                         max_iter=n_epochs,
                         verbose=verbose,
                         early_stopping=True,
                         n_iter_no_change=patience,
    )


    print('Generating training examples ...')
    if sample_points is None:
        sample_points = sample_domain(bounds, n_training_pts)
        print(sample_points.shape)
    sample_ssps = encoder.encode(sample_points)
    
    if corrupt_training_examples == True:
        if gaussian_noise_std == 'auto':
            gaussian_noise_std = 1/np.sqrt(encoder.ssp_dim)
        sample_ssps += np.random.normal(loc = 0., scale = gaussian_noise_std, size = sample_ssps.shape) 

#     shuffled_ssps, shuffled_pts = sklearn.utils.shuffle(sample_ssps, sample_points)

    print('Training decoder network ...')
    model.fit(sample_ssps, sample_points)

    history = {
            'loss': model.loss_curve_,
            'val_loss': model.validation_scores_,
            }
        
    return SSPDecoder(bounds, model, encoder), history


def train_decoder_net_tf(encoder, bounds, n_training_pts=200000
                      hidden_units = [8],
                      learning_rate=1e-3, 
                      n_epochs = 100, 
                      patience=3,
                      verbose=True):
    '''
        Trains a dense neural network to decode SSPs.

    '''


    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')

    import sklearn
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
        

    architecture = [encoder.ssp_dim] + hidden_units 
    arch_layers = [layers.Dense(a,activation='relu') for a in architecture]
    arch_layers += [layers.Dense(bounds.shape[0], name='output')]
    model = keras.Sequential(arch_layers)



    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error')

    sample_points = sample_domain(bounds, n_training_pts)
    sample_ssps = encoder.encode(sample_points)

    shuffled_ssps, shuffled_pts = sklearn.utils.shuffle(sample_ssps, sample_points)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    history = model.fit(shuffled_ssps, shuffled_pts, 
                        epochs=n_epochs,
                        verbose=verbose, 
                        callbacks=[callback],
                        validation_split = 0.1)
        
    return SSPDecoder(bounds, model, encoder), history
