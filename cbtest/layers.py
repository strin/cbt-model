import theano
import theano.tensor as T

floatX = theano.config.floatX
log = T.log
dot = T.dot
mean = T.mean
softmax = T.nnet.softmax

def stack(*tensors):
    return T.stack(tensors, axis=0)

class Embed(object):
    '''
    embed discrete objects into vector space.
    '''
    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.W = theano.shared(value=(npr.randn(vocab_size, hidden_dim) * 0.01).astype(),
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



