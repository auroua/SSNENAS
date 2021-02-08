import numpy as np
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import clear_session


def mle_loss(y_true, y_pred):
    # Minimum likelihood estimate loss function
    mean = tf.slice(y_pred, [0, 0], [-1, 1])
    var = tf.slice(y_pred, [0, 1], [-1, 1])
    return 0.5 * tf.log(2*np.pi*var) + tf.square(y_true - mean) / (2*var)


def mape_loss(y_true, y_pred):
    # Minimum absolute percentage error loss function
    lower_bound = 4.5
    fraction = tf.math.divide(tf.subtract(y_pred, lower_bound),
                              tf.subtract(y_true, lower_bound))
    return tf.abs(tf.subtract(fraction, 1))


class PredictorMLP:
    def __init__(self, gpu='0', sess=None):
        self.sess = sess
        if not isinstance(gpu, str):
            gpu = str(gpu)
        self.gpu = gpu

    def clear_gpu(self):
        config = tf.ConfigProto(device_count={'GPU': 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.gpu)
        self.sess = tf.Session(config=config)
        clear_session()

    def get_dense_model(self,
                        input_dims,
                        num_layers,
                        layer_width,
                        loss,
                        regularization,
                        dynamic_filter=0,
                        dynamic_kernel=0):
        config = tf.ConfigProto(device_count={'GPU': 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.gpu)
        self.sess = tf.Session(config=config)
        keras.backend.set_session(self.sess)

        input_layer = keras.layers.Input(input_dims)
        model = keras.models.Sequential()

        # if dynamic_kernel > 0 and dynamic_filter > 0:
        #     model.add(keras.layers.Conv1D(filters=dynamic_filter, kernel_size=dynamic_kernel))

        # for _ in range(num_layers):
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(2048, activation='relu'))
        model.add(keras.layers.Dense(2048, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))

        model = model(input_layer)
        if loss == 'mle':
            mean = keras.layers.Dense(1)(model)
            var = keras.layers.Dense(1)(model)
            var = keras.layers.Activation(tf.math.softplus)(var)
            output = keras.layers.concatenate([mean, var])
        else:
            if regularization == 0:
                output = keras.layers.Dense(1)(model)
            else:
                reg = keras.regularizers.l1(regularization)
                output = keras.layers.Dense(1, kernel_regularizer=reg)(model)

        dense_net = keras.models.Model(inputs=input_layer, outputs=output)
        return dense_net

    def fit(self, xtrain, ytrain,
            num_layers=10,
            layer_width=20,
            loss='mae',
            epochs=200,
            batch_size=32,
            lr=.01,
            verbose=0,
            regularization=0,
            dynamic_filter=0,
            dynamic_kernel=0):

        if loss == 'mle':
            loss_fn = mle_loss
        elif loss == 'mape':
            loss_fn = mape_loss
        else:
            loss_fn = 'mae'

        self.model = self.get_dense_model((xtrain.shape[1],),
                                          loss=loss_fn,
                                          num_layers=num_layers,
                                          layer_width=layer_width,
                                          regularization=regularization,
                                          dynamic_filter=dynamic_filter,
                                          dynamic_kernel=dynamic_kernel)
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=.9, beta_2=.99)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        self.model.fit(xtrain, ytrain,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose)
        train_pred = np.squeeze(self.model.predict(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error

    def predict(self, xtest):
        return self.model.predict(xtest)
