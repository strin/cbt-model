from common import *

log = T.log
dot = T.dot
mean = T.mean
softmax = T.nnet.softmax
tanh = T.tanh
sigmoid = T.nnet.sigmoid


_position_encoding = {}
def position_encoding(senlen, dim):
    '''
    the position weighting scheme used in 
        Sainbayar et al. 2015. end-to-end memory networks.

    l_kj = (1-j/J) - (k/d) * (1 - 2j / J)
    '''
    global _position_encoding
    if _position_encoding.get((senlen, dim)) == None:
        lmat = np.zeros((1, senlen, dim), dtype=floatX)
        J = float(senlen)
        D = float(dim)
        for j in range(senlen):
            for k in range(dim):
                lmat[0, j, k] = (1. - (j + 1) / J) - \
                        ((k + 1) / D) * (1. - 2 * (j + 1) / J)
        _position_encoding[(senlen, dim)] = theano.shared(lmat, broadcastable=(True, False, False))
    return _position_encoding[(senlen, dim)]


def stack(*tensors):
    return T.stack(tensors, axis=0)


def ortho_weight(ndim):
    '''
    generate orthogonal weights (row vectors)
    '''
    W = npr.randn(ndim, ndim)
    u, s, v = npla.svd(W)
    return u.astype(theano.config.floatX)


class Layer(object):
    def __call__(self):
        raise NotImplementedError('__call__ not implemented for layer')


    def get_params(self):
        return {param.name: param.get_value() for param in self.params}


    def set_params(self, **params):
        for param in self.params:
            if param.name in params:
                param.set_value(params[param.name])



class Embed(Layer):
    '''
    embed discrete objects into vector space.
    '''
    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.W = theano.shared(value=(npr.randn(vocab_size, hidden_dim) * 0.01).astype(theano.config.floatX),
                               name='W')
        self.params = [self.W]


    def __call__(self, symbols):
        return self.W[symbols, :]


class LSTM(Layer):
    '''
    basic LSTM layer
    '''
    def __init__(self, batchsize, hidden_dim):
        self.hidden_dim = hidden_dim
        self.batchsize = batchsize
        self.W = theano.shared(
                    np.concatenate([ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim)], axis=1),
                    name='W'
                )

        self.U = theano.shared(
                    np.concatenate([ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim)], axis=1),
                    name='U'
                )
        self.b = theano.shared(
                    np.zeros((1, 4 * hidden_dim), dtype=theano.config.floatX),
                    name='b',
                    broadcastable=(True, False)
                )
        self.params = [self.W, self.U, self.b]


    def _step(self, x, c, h):
        W = lambda i: self.W[:, self.hidden_dim * (i-1) : self.hidden_dim * i]
        U = lambda i: self.U[:, self.hidden_dim * (i-1) : self.hidden_dim * i]
        b = lambda i: self.b[:, self.hidden_dim * (i-1) : self.hidden_dim * i]
        i = sigmoid(dot(x, W(1)) + dot(h, U(1)) + b(1))
        f = sigmoid(dot(x, W(2)) + dot(h, U(2)) + b(2))
        o = sigmoid(dot(x, W(3)) + dot(h, U(3)) + b(3))
        _c = tanh(dot(x, W(4)) + dot(h, U(4)) + b(4))

        c = _c * i + c * f
        h = o * tanh(c)

        return c, h


    def __call__(self, inputs):
        zero = np.array([0.], dtype=theano.config.floatX)[0]
        result, updates = theano.scan(lambda x, c, h: self._step(x, c, h),
                                      outputs_info = [T.zeros((self.batchsize, self.hidden_dim)),
                                                    T.zeros((self.batchsize, self.hidden_dim))
                                                      ],
                                        name='lstm_layer',
                                        sequences=[inputs]
                    )
        return result[1][-1] # return h.



class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        self.W = theano.shared(value=(npr.randn(input_dim, output_dim) * 0.01).astype(floatX),
                               name='W')
        self.b = theano.shared(value=(np.zeros(output_dim, dtype=floatX)), name='b')
        self.params = [self.W, self.b]


    def __call__(self, xs):
        return dot(xs, self.W) + self.b


class MemoryLayer(object):
    def __init__(self, batchsize, mem_size, sen_maxlen, vocab_size, hidden_dim,
                 encoder='bow', flags={
                        'position_encoding': False
                     }):
        self.batchsize = batchsize
        self.mem_size = mem_size
        self.sen_maxlen = sen_maxlen
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.params = []

        self.input_embed = Embed(vocab_size, hidden_dim)
        self.output_embed = Embed(vocab_size, hidden_dim)
        self.params.extend(self.input_embed.params)
        self.params.extend(self.output_embed.params)

        if encoder == 'bow':
            if flags['position_encoding']:
                print '[memory layer] use PE'
                lmat = position_encoding(self.sen_maxlen, self.hidden_dim)
                self.encoder_func = lambda m: mean(m * lmat, axis=2)
            else:
                self.encoder_func = lambda m: mean(m, axis=2)
        elif encoder == 'lexical':
            lmat = position_encoding(self.sen_maxlen, self.hidden_dim)
            self.encoder_func = lambda m: (m * lmat).dimshuffle(0, 3, 1, 2).flatten(3).dimshuffle(0, 2, 1)
        elif encoder == 'lstm':
            lstm = LSTM(self.batchsize, self.hidden_dim)
            self.params.extend(lstm.params)
            def lstm_encode(m):
                results, updates = theano.scan(fn=lambda mi: lstm(mi),
                                               sequences=[m.dimshuffle(1, 2, 0, 3)]
                                               )
                return results.dimshuffle(1, 0, 2)
            self.encoder_func = lambda m: lstm_encode(m)
        elif encoder == 'weighted':
            linear = LinearLayer(self.sen_maxlen, 1)
            self.params.extend(linear.params)
            self.encoder_func = lambda m: linear(m.dimshuffle(0, 1, 3, 2)).flatten(3)


    def __call__(self, contexts, u):
        '''
        assume #context = mem_size

        contexts has shape (batchsize, #context, seqlen)
        u has shape (batchsize, hidden_dim)
        '''
        # memory vectors.
        m = T.reshape(self.input_embed(contexts.flatten()), (self.batchsize, self.mem_size, self.sen_maxlen, self.hidden_dim))
        m = self.encoder_func(m)
        # output vectors.
        c = T.reshape(self.output_embed(contexts.flatten()), (self.batchsize, self.mem_size, self.sen_maxlen, self.hidden_dim))
        c = self.encoder_func(c)

        probs, updates = theano.scan(fn=lambda mvs, uv: softmax(dot(mvs, uv)),
                        sequences=[m, u]
                        )

        outputs, updates = theano.scan(fn=lambda probv, cv: dot(cv, T.transpose(probv)),
                                       sequences=[probs, c.dimshuffle(0, 2, 1)]
                                       )

        outputs = outputs.flatten(2) # turn row vector into vector.
        return outputs








