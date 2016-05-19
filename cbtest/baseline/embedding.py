# basic embeding models as baselines.
from cbtest.common import *
from cbtest.layers import (log, dot, mean, softmax, Embed, floatX, stack, LSTM, MemoryLayer, LinearLayer, position_encoding)
from cbtest.utils import choice, Timer
from cbtest.evaluate import (accuracy, disagree)
from cbtest.dataset import remove_stopwords, lower, remove_punctuation, filter, unkify
import cbtest.optimizers as optimizers

import theano.tensor as T


class CBTLearner(object):
    def __init__(self):
        self.num_context = 20
        self.num_candidate = 10
        self.sen_maxlen = 128


    def compile(exs):
        pass


    def train(exs, num_iter=100):
        pass


    def test(exs):
        pass


class BowEmbedLearner(CBTLearner):
    def __init__(self, batchsize=1, hidden_dim=100, lr=1e-4, sen_maxlen=128, flags={}, **kwargs):
        CBTLearner.__init__(self)
        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.flags = {}
        self.kwargs = kwargs


    def preprocess_sentence(self, sentence):
        '''
        preprocess training sentences.
        '''
        return lower(sentence)


    def preprocess_dataset(self, exs):
        '''
        preprocess test sentences.
        '''
        def preprocess(sentence):
            sen = unkify(self.preprocess_sentence(sentence), self.vocab)
            if len(sen) > self.sen_maxlen:
                print '[warning] exceeding sentence max length.'
                sen = sen[:self.sen_maxlen]
            return sen
        for ex in exs:
            new_context = []
            for sen in ex['context']:
                sen = preprocess(sen)
                new_context.append(sen)
            ex['context'] = new_context
            ex['query'] = preprocess(ex['query'])
            ex['candidate'] = preprocess(ex['candidate'])
            ex['answer'] = preprocess([ex['answer']])[0]


    def create_vocab(self, exs):
        vocab = {
            '<null>': 0, # no word in current position.
            '<unk>': 1 # unknown word.
        }

        def add_word_if_not_exist(word):
            if word not in vocab:
                vocab[word] = len(vocab)

        def add_sen_if_not_exist(sentence):
            sentence = self.preprocess_sentence(sentence)
            for word in sentence:
                add_word_if_not_exist(word)
            if len(sentence) > self.sen_maxlen:
                self.sen_maxlen = len(sentence)

        for ex in exs:
            for sen in ex['context']:
                add_sen_if_not_exist(sen)
            add_sen_if_not_exist(ex['query'])
            add_sen_if_not_exist(ex['candidate'])

        vocab_size = len(vocab)
        print '[vocab size]', vocab_size
        print '[max sentence length]', self.sen_maxlen
        print '[num train exs]', len(exs)
        print '[batchsize]', self.batchsize

        self.vocab = vocab
        self.vocab_size = vocab_size


    def compile(self, train_exs):
        exs = train_exs

        # first pass, create vocab.
        self.create_vocab(exs)

        # build neural net.
        contexts = T.ltensor3('contexts')
        querys = T.ltensor3('querys')
        yvs = T.lvector('yvs')
        cvs = T.lmatrix('cvs')

        probs = []
        params = []

        embed = Embed(self.vocab_size, self.hidden_dim)
        params.extend(embed.params)

        c_emb = T.reshape(embed(contexts.flatten()), (self.batchsize, self.num_context, self.sen_maxlen, self.hidden_dim)) # 'C', row-major.
        c_emb = mean(c_emb, axis=2) # average over words.
        c_emb = mean(c_emb, axis=1) # average over sentences.
        c_emb = c_emb.dimshuffle(0, 'x', 1)

        #querys = querys[:, 1:, :] # take candidate querys.
        q_emb = T.reshape(embed(querys[:, 1:, :].flatten()), (self.batchsize, self.num_candidate, self.sen_maxlen, self.hidden_dim)) # 'C', row-major.
        q_emb = mean(q_emb, axis=2)

        scores = T.sum(c_emb * q_emb, axis=2) # batchsize x num_candidate
        probs = softmax(scores)

        outputs = T.zeros((self.batchsize, self.vocab_size), dtype=floatX)
        outputs = T.set_subtensor(outputs[T.transpose(stack(*[T.arange(self.batchsize)] * self.num_candidate)),
                                          cvs], probs)
        loss = -mean(log(outputs[T.arange(self.batchsize), yvs]))

        self.fprop = theano.function(inputs=[contexts, querys, cvs], outputs=outputs)

        updates = optimizers.Adam(loss, params, alpha=FX(self.lr))
        self.bprop = theano.function(inputs=[contexts, querys, cvs, yvs],
                                        outputs=loss, updates=updates)


    def train(self, exs, num_iter=100):
        self.preprocess_dataset(exs)

        # second pass, learn embedding.
        def sample_minibatch():
            minibatch = choice(exs, size=self.batchsize, replace=True)
            meta = {}
            meta['minibatch'] = minibatch
            (contexts, querys, yvs, cvs) = self.encode_minibatch(minibatch)

            return (contexts, querys, yvs, cvs, meta)

        for it in range(num_iter):
            for ni in range(len(exs) / self.batchsize + 1):
                (contexts, querys, yvs, cvs, meta) = sample_minibatch()

                error = self.bprop(contexts, querys, cvs, yvs)
                if ni % 10 == 0:
                    print 'iter', ni * self.batchsize, '/', len(exs), 'error', error


    def encode_minibatch(self, minibatch):
        '''
        encode minibatch of context + query into vector representation.
        '''
        contexts = []
        querys = []
        yvs = []
        cvs = []
        for ex in minibatch:
            # encode context.
            context = np.zeros((self.num_context, self.sen_maxlen), dtype=np.int64)
            for (si, sen) in enumerate(ex['context']):
                for (wi, word) in enumerate(sen):
                    context[si, wi] = self.vocab[word]
            contexts.append(context)

            # encode query.
            # first line will be original query with xxx.
            # each line that follows is the query with each candidate word
            # plugged in.
            query = np.zeros((len(ex['candidate']) + 1, self.sen_maxlen), dtype=np.int64)
            qv = np.array([self.vocab[word] for word in ex['query']], dtype=np.int64)
            query[0, :len(qv)] = qv
            blank = ex['query'].index('xxxxx')
            for (ci, candidate) in enumerate(ex['candidate']):
                query[ci + 1, :len(qv)] = qv
                query[ci + 1, blank] = self.vocab[candidate]
            querys.append(query)

            yvs.append(self.vocab[ex['answer']])

            cv = [self.vocab[word] for word in ex['candidate']]
            cvs.append(cv)

        contexts = np.array(contexts, dtype=np.int64)
        querys = np.array(querys, dtype=np.int64)
        yvs = np.array(yvs, dtype=np.int64)
        cvs = np.array(cvs, dtype=np.int64)

        return (contexts, querys, yvs, cvs)


    def test(self, exs):
        self.preprocess_dataset(exs)

        all_preds = []
        all_truths = []
        for offset in range(0, len(exs), self.batchsize):
            minibatch = exs[offset:offset + self.batchsize]
            while len(minibatch) < self.batchsize:
                minibatch.append(minibatch[-1])
            (contexts, querys, yvs, cvs) = self.encode_minibatch(minibatch)
            yvs = np.array(yvs).astype(theano.config.floatX)
            inds = np.argmax(self.fprop(contexts, querys, cvs)
                              [np.transpose([range(self.batchsize)] * self.num_candidate),
                               cvs], axis=1)
            preds = cvs[range(self.batchsize), inds]
            truths = np.array(yvs, dtype=np.int64)
            if sum(truths == 1):
                print '[warning] unknown word in answer'
            truths[truths == 1] = -1
            all_preds.extend(preds[:min(self.batchsize, len(exs)-offset)])
            all_truths.extend(truths[:min(self.batchsize, len(exs)-offset)])
        acc = accuracy(all_preds, all_truths)
        errs = disagree(all_preds, all_truths)
        return (acc, errs)


class LSTMContextQuery(BowEmbedLearner):
    def __init__(self, batchsize=1, hidden_dim=100, lr=1e-4):
        CBTLearner.__init__(self)
        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr


    def compile(self, train_exs):
        # build vocabulary.
        exs = train_exs
        self.create_vocab(exs)

        # build LSTM encoder.
        # build neural net.
        contexts = T.ltensor3('contexts')
        querys = T.ltensor3('querys')
        yvs = T.lvector('yvs')
        cvs = T.lmatrix('cvs')

        params = []

        embed = Embed(self.vocab_size, self.hidden_dim)
        lstm = LSTM(self.batchsize, self.hidden_dim)
        linear = LinearLayer(self.hidden_dim, self.vocab_size)

        params.extend(embed.params)
        params.extend(lstm.params)
        params.extend(linear.params)

        context_word_embs = T.reshape(embed(contexts.flatten()), (self.batchsize, self.num_context * self.sen_maxlen, self.hidden_dim))
        query_word_embs = T.reshape(embed(querys[:, 0:1, :].flatten()), (self.batchsize, self.sen_maxlen, self.hidden_dim)) # 'C', row-major.
        context_query_word_embs = T.concatenate((context_word_embs, query_word_embs), axis=1)
        context_query_word_embs = context_query_word_embs.dimshuffle(1, 0, 2)
        all_embs = lstm(context_query_word_embs)
        probs = softmax(linear(all_embs))

        loss = -mean(log(probs[T.arange(self.batchsize), yvs]))


        print '[compiling forward prop]'
        self.fprop = theano.function(inputs=[contexts, querys, cvs], outputs=probs,
                                     on_unused_input='ignore')

        print '[compiling backward prop]'
        updates = optimizers.Adam(loss, params, alpha=FX(self.lr))
        self.bprop = theano.function(inputs=[contexts, querys, cvs, yvs],
                                        outputs=loss, updates=updates,
                                     on_unused_input='ignore')


class MemoryNetwork(BowEmbedLearner):
    def __init__(self, batchsize=1, hidden_dim=100, hop=1, lr=1e-4,
                 flags={
                     'position_encoding': False
                 }, **kwargs):
        BowEmbedLearner.__init__(self, **kwargs)
        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hop = hop
        self.kwargs = kwargs

        self.flags = flags # use time/order encoding.

        #self.preprocess_train = lambda sentence:\
        #    remove_stopwords(remove_punctuation(lower(sentence)))

        self.preprocess = lambda sentence: lower(sentence) # less aggressive.


    def compile(self, train_exs):

        exs = train_exs
        self.create_vocab(exs)

        contexts = T.ltensor3('contexts')
        querys = T.ltensor3('querys')
        yvs = T.lvector('yvs')
        cvs = T.lmatrix('cvs')

        params = []

        question_layer = Embed(self.vocab_size, self.hidden_dim)
        q = T.reshape(question_layer(querys[:, 0:1, :].flatten()),
                      (self.batchsize, self.sen_maxlen, self.hidden_dim)
                      )
        lmat = position_encoding(self.sen_maxlen, self.hidden_dim)
        if self.flags['position_encoding']:
            print '[memory network] use PE'
            q = q * lmat
        u = mean(q, axis=1)
        params.extend(question_layer.params)

        mem_layers = []
        for hi in range(self.hop):
            mem_layer = MemoryLayer(self.batchsize, self.num_context, self.sen_maxlen, self.vocab_size, self.hidden_dim, flags=self.flags,
                                    **self.kwargs)
            params.extend(mem_layer.params)
            mem_layers.append(mem_layer)
            o = mem_layer(contexts, u)
            u = u + o

        linear = LinearLayer(self.hidden_dim, self.vocab_size)
        params.extend(linear.params)

        print '[memory network]', params

        probs = softmax(linear(u))

        loss = -mean(log(probs[T.arange(self.batchsize), yvs]))

        print '[compiling forward prop]'
        self.fprop = theano.function(inputs=[contexts, querys, cvs], outputs=probs,
                                     on_unused_input='ignore')

        print '[compiling backward prop]'
        updates = optimizers.Adam(loss, params, alpha=FX(self.lr))
        self.bprop = theano.function(inputs=[contexts, querys, cvs, yvs],
                                        outputs=loss, updates=updates,
                                     on_unused_input='ignore')


