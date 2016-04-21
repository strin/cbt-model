from common import *

floatX = theano.config.floatX
log = T.log
dot = T.dot
mean = T.mean
softmax = T.nnet.softmax
tanh = T.tanh
sigmoid = T.nnet.sigmoid

def stack(*tensors):
    return T.stack(tensors, axis=0)


def ortho_weight(ndim):
    '''
    generate orthogonal weights (row vectors)
    '''
    W = npr.randn(ndim, ndim)
    u, s, v = npla.svd(W)
    return np.transpose(u.astype(theano.config.floatX))


class Embed(object):
    '''
    embed discrete objects into vector space.
    '''
    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.W = theano.shared(value=(npr.randn(vocab_size, hidden_dim) * 0.01).astype(theano.config.floatX),
                               name='W')
        self.params = [self.W]


    def get_params(self):
        return {param.name: param.get_value() for param in self.params}


    def set_params(self, **params):
        for param in self.params:
            if param.name in params:
                param.set_value(params[param.name])


    def __call__(self, symbols):
        return self.W[symbols, :]


class LSTM(object):
    '''
    basic LSTM layer
    '''
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.W = theano.shared(
                    np.concatenate([ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim)], axis=0),
                    name='W'
                )

        self.U = theano.shared(
                    np.concatenate([ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim),
                                    ortho_weight(hidden_dim)], axis=0),
                    name='U'
                )
        self.b = theano.shared(
                    np.zeros(4 * hidden_dim),
                    name='b'
                )

    def _step(self, x, c, h):
        W = lambda i: self.W[self.hidden_dim * (i-1) : self.hidden_dim * i, :]
        U = lambda i: self.U[self.hidden_dim * (i-1) : self.hidden_dim * i, :]
        b = lambda i: self.b[self.hidden_dim * (i-1) : self.hidden_dim * i]
        i = sigmoid(dot(W(1), x) + dot(U(1), h) + b(1))
        f = sigmoid(dot(W(2), x) + dot(U(2), h) + b(2))
        o = sigmoid(dot(W(3), x) + dot(U(3), h) + b(3))
        _c = tanh(dot(W(4), x) + dot(U(4), h) + b(4))

        c = _c * i + c * f
        h = o * tanh(c)

        return c, h


    def __call__(self, inputs):
        zero = np.array([0.], dtype=theano.config.floatX)[0]
        result, updates = theano.scan(lambda x, c, h: self._step(x, c, h),
                                      outputs_info = [T.alloc(zero, self.hidden_dim),
                                                    T.alloc(zero, self.hidden_dim),
                                                      ],
                                        name='lstm_layer',
                                        sequences=[inputs]
                    )
        return result[1][-1] # return h.







