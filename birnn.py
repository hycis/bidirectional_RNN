import theano
import theano.tensor as T

from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
from keras.layers.core import Layer
import numpy as np


class Transform(Layer):
     '''
         This function is needed transform the dataset such that we can
         add mlp layers before RNN/LSTM or after the RNN/LSTM.
     '''
     def __init__(self, dims, input=None):
         '''
         If input is three dimensional tensor3, with dimensions (nb_samples, sequence_length, vector_length)
         and dims is tuple (vector_length,), then the output will be (nb_samples * sequence_length, vector_length)
         If input is two dimensional matrix, with dimensions (nb_samples * sequence_length, vector_length)
         and if we want to revert back to (nb_samples, sequence_length, vector_length) so that we can feed
         the LSTM, then we can set dims as (sequence_length, vector_length).
         This function is needed for adding mlp layers before LSTM or after the LSTM.
         When used as first layer, input has to be set either as tensor3 or matrix
         '''

         super(Transform, self).__init__()
         self.dims = dims
         if input is not None:
             self.input = input

     def get_output(self, train):
         X = self.get_input(train)
         first_dim = T.prod(X.shape) / np.prod(self.dims)
         return T.reshape(X, (first_dim,)+self.dims)

     def get_config(self):
         return {"name":self.__class__.__name__,
             "dims":self.dims}


class BiDirectionLSTM(Layer):
    '''
        Acts as a spatiotemporal projection,

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
        init='glorot_uniform', inner_init='orthogonal',
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, output_mode='sum', return_sequences=False):

        super(BiDirectionLSTM,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.output_mode = output_mode # output_mode is either sum or concatenate
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        # forward weights
        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_zeros((self.output_dim))

        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        # backward weights
        self.Wb_i = self.init((self.input_dim, self.output_dim))
        self.Ub_i = self.inner_init((self.output_dim, self.output_dim))
        self.bb_i = shared_zeros((self.output_dim))

        self.Wb_f = self.init((self.input_dim, self.output_dim))
        self.Ub_f = self.inner_init((self.output_dim, self.output_dim))
        self.bb_f = shared_zeros((self.output_dim))

        self.Wb_c = self.init((self.input_dim, self.output_dim))
        self.Ub_c = self.inner_init((self.output_dim, self.output_dim))
        self.bb_c = shared_zeros((self.output_dim))

        self.Wb_o = self.init((self.input_dim, self.output_dim))
        self.Ub_o = self.inner_init((self.output_dim, self.output_dim))
        self.bb_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,

            self.Wb_i, self.Ub_i, self.bb_i,
            self.Wb_c, self.Ub_c, self.bb_c,
            self.Wb_f, self.Ub_f, self.bb_f,
            self.Wb_o, self.Ub_o, self.bb_o,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _forward_step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):
        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        g_t = self.activation(xc_t + T.dot(h_tm1, u_c))
        c_t = f_t * c_tm1 + i_t * g_t
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_forward_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._forward_step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient
        )
        return outputs.dimshuffle((1,0,2))


    def get_backward_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.Wb_i) + self.bb_i
        xf = T.dot(X, self.Wb_f) + self.bb_f
        xc = T.dot(X, self.Wb_c) + self.bb_c
        xo = T.dot(X, self.Wb_o) + self.bb_o

        [outputs, memories], updates = theano.scan(
            self._forward_step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.Ub_i, self.Ub_f, self.Ub_o, self.Ub_c],
            go_backwards = True,
            truncate_gradient=self.truncate_gradient
        )
        return outputs.dimshuffle((1,0,2))


    def get_output(self, train):
        forward = self.get_forward_output(train)
        backward = self.get_backward_output(train)
        if self.output_mode is 'sum':
            output = forward + backward
        elif self.output_mode is 'concat':
            output = T.concatenate([forward, backward], axis=2)
        else:
            raise Exception('output mode is not sum or concat')
        if self.return_sequences==False:
            return output[:,-1,:]
        elif self.return_sequences==True:
            return output
        else:
            raise Exception('Unexpected output shape for return_sequences')


    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient}
