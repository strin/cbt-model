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
        lmat = np.zeros((senlen, dim), dtype=floatX)
        J = float(senlen)
        D = float(dim)
        for j in range(senlen):
            for k in range(dim):
                lmat[j, k] = (1. - (j + 1) / J) - \
                        ((k + 1) / D) * (1. - 2 * (j + 1) / J)
        _position_encoding[(senlen, dim)] = theano.shared(lmat, broadcastable=(False, False))
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
    def __init__(self, hidden_dim=100):
        self.hidden_dim = hidden_dim
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
        dot = lambda a, b: T.tensordot(a, b, axes=1) # use more generalized tensordot.
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


    def __call__(self, inputs, n_steps=None):
        result, updates = theano.scan(lambda x, c, h: self._step(x, c, h),
                        outputs_info = [T.zeros_like(inputs[0]),
                                        T.zeros_like(inputs[0])
                                                      ],
                                        name='lstm_layer',
                                        sequences=[inputs],
                                        n_steps=n_steps
                    )
        return result[1] # return h.


class LSTMq(Layer):
    '''
    adaptive LSTM
    '''
    def __init__(self, batchsize, hidden_dim):
        self.hidden_dim = hidden_dim
        self.batchsize = batchsize
        self.W = theano.shared(
                    np.concatenate([ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
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
                    np.zeros((1, 4 * hidden_dim * 2), dtype=theano.config.floatX),
                    name='b',
                    broadcastable=(True, False)
                )
        self.params = [self.W, self.U, self.b]


    def _step(self, x, c, h, q):
        W = lambda i: self.W[:, self.hidden_dim * (i-1) : self.hidden_dim * i]
        U = lambda i: self.U[:, self.hidden_dim * (i-1) : self.hidden_dim * i]
        b = lambda i: self.b[:, self.hidden_dim * (i-1) : self.hidden_dim * i]
        i = sigmoid(dot(x, W(1)) + dot(q, W(2)) + dot(h, U(1)) + b(1))
        f = sigmoid(dot(x, W(3)) + dot(q, W(4)) + dot(h, U(2)) + b(2))
        o = sigmoid(dot(x, W(5)) + dot(q, W(6)) + dot(h, U(3)) + b(3))
        _c = tanh(dot(x, W(7)) + dot(h, U(4)) + b(4))

        c = _c * i + c * f
        h = o * tanh(c)

        return c, h


    def __call__(self, inputs, query):
        zero = np.array([0.], dtype=theano.config.floatX)[0]
        result, updates = theano.scan(self._step,
                                      outputs_info = [T.zeros((self.batchsize, self.hidden_dim)),
                                                    T.zeros((self.batchsize, self.hidden_dim))
                                                      ],
                                        name='lstm_layer',
                                        sequences=[inputs],
                                        non_sequences=query
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
    def __init__(self, batchsize, mem_size, unit_size, vocab_size, hidden_dim,
                 encoder='bow', **kwargs):
        self.batchsize = batchsize
        self.mem_size = mem_size
        self.unit_size = unit_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.params = []
        self.kwargs = kwargs

        self.input_embed = Embed(vocab_size, hidden_dim)
        self.output_embed = Embed(vocab_size, hidden_dim)
        self.params.extend(self.input_embed.params)
        self.params.extend(self.output_embed.params)

        if encoder == 'bow':
            if kwargs['position_encoding']:
                print '[memory layer] use PE'
                lmat = position_encoding(self.unit_size, self.hidden_dim).dimshuffle('x', 'x', 0, 1)
                self.encoder_func = lambda m: mean(m * lmat, axis=2)
            else:
                self.encoder_func = lambda m: mean(m, axis=2)
        elif encoder == 'lstm':
            print 'using LSTM encoder'
            lstm = LSTM(self.batchsize, self.hidden_dim)
            self.params.extend(lstm.params)
            self.encoder_func = lambda m: lstm(m.dimshuffle(2, 0, 1, 3))
        elif encoder == 'lstm2': # bidirectional lstm towards center.
            print 'using LSTM2 encoder'
            lstm1 = LSTM(self.batchsize, self.hidden_dim)
            self.params.extend(lstm1.params)
            lstm2 = LSTM(self.batchsize, self.hidden_dim)
            self.params.extend(lstm2.params)
            self.encoder_func = lambda m: (lstm1(m.dimshuffle(2, 0, 1, 3), n_steps=int(np.ceil(unit_size/2)))
                                          +lstm2(m.dimshuffle(2, 0, 1, 3)[::-1, :, :, :], n_steps=int(np.ceil(unit_size/2)))) / floatX(2.0)
        elif encoder == 'lstm2-shared': # bidirectional lstm towards center.
            print 'using LSTM2 shared encoder'
            lstm = LSTM(self.batchsize, self.hidden_dim)
            self.params.extend(lstm.params)
            self.encoder_func = lambda m: (lstm(m.dimshuffle(2, 0, 1, 3), n_steps=int(np.ceil(unit_size/2)))
                                          +lstm(m.dimshuffle(2, 0, 1, 3)[::-1, :, :, :], n_steps=int(np.ceil(unit_size/2)))) / floatX(2.0)

        elif encoder == 'weighted':
            linear = LinearLayer(self.unit_size, 1)
            self.params.extend(linear.params)
            self.encoder_func = lambda m: linear(m.dimshuffle(0, 1, 3, 2)).flatten(3)


    def get_probs(self, contexts, u):
        # memory vectors.
        m = T.reshape(self.input_embed(contexts.flatten()), (self.batchsize, self.mem_size, self.unit_size, self.hidden_dim))
        m = self.encoder_func(m)
        probs, updates = theano.scan(fn=lambda mvs, uv: softmax(dot(mvs, uv)),
                        sequences=[m, u]
                        )
        return probs

    def __call__(self, contexts, u):
        '''
        assume #context = mem_size

        contexts has shape (batchsize, #context, seqlen)
        u has shape (batchsize, hidden_dim)
        '''
        probs = self.get_probs(contexts, u)

        # output vectors.
        c = T.reshape(self.output_embed(contexts.flatten()), (self.batchsize, self.mem_size, self.unit_size, self.hidden_dim))
        c = self.encoder_func(c)

        outputs, updates = theano.scan(fn=lambda probv, cv: dot(cv, T.transpose(probv)),
                                       sequences=[probs, c.dimshuffle(0, 2, 1)]
                                       )

        outputs = outputs.flatten(2) # turn row vector into vector.
        return outputs








