# basic embeding models as baselines.
from cbtest.common import *
from cbtest.layers import (log, dot, mean, softmax, Embed, floatX, stack, LSTM, MemoryLayer)
from cbtest.utils import choice
from cbtest.evaluate import (accuracy, disagree)
from cbtest.dataset import remove_stopwords, lower, remove_punctuation, filter
import cbtest.optimizers as optimizers

import theano.tensor as T

class BowEmbedLearner(object):
    def __init__(self, batchsize=1, hidden_dim=100, lr=1e-4):
        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr

        #self.preprocess = lambda sentence:\
        #    remove_stopwords(remove_punctuation(lower(sentence)))

        self.preprocess = lambda sentence: lower(sentence) # less aggressive.


    def create_vocab(self, exs):
        vocab = {}
        def add_word_if_not_exist(word):
            if word not in vocab:
                vocab[word] = len(vocab)

        for ex in exs:
            for sen in ex['context']:
                for word in self.preprocess(sen):
                    add_word_if_not_exist(word)
            for word in self.preprocess(ex['query']):
                add_word_if_not_exist(word)
            for word in self.preprocess(ex['candidate']):
                add_word_if_not_exist(word)

        vocab_size = len(vocab)
        print '[vocab size]', vocab_size
        print '[num train exs]', len(exs)
        print '[batchsize]', self.batchsize
        self.vocab = vocab
        self.vocab_size = vocab_size


    def compile(self, train_exs):
        exs = train_exs

        # first pass, create vocab.
        self.create_vocab(exs)

        # build neural net.
        xs = []
        probs = []
        ys = T.matrix('ys')
        params = []

        embed = Embed(self.vocab_size, self.hidden_dim)
        params.extend(embed.params)

        for bi in range(self.batchsize):
            c = T.lvector('context_' + str(bi))
            c_emb = mean(embed(c), axis=0)
            qs = []
            scores = []
            for can_id in range(10):
                q = T.lvector('query_' + str(bi) + '_' + str(can_id))
                q_emb = mean(embed(q), axis=0)
                score = dot(c_emb, q_emb)
                qs.append(q)
                scores.append(score)
            score_vector = stack(*scores)
            prob = softmax(score_vector)

            xs.append([c] + qs)
            probs.append(prob)

        probs = stack(*probs)
        loss = -mean(log(probs) * ys)

        self.fprop = theano.function(inputs=sum(xs, []), outputs=probs)

        updates = optimizers.Adam(loss, params, alpha=FX(self.lr))
        self.bprop = theano.function(inputs=sum(xs, []) + [ys],
                                        outputs=loss, updates=updates)


    def train(self, exs, num_iter=100):
        # second pass, learn embedding.
        def sample_minibatch():
            minibatch = choice(exs, size=self.batchsize, replace=True)
            meta = {}
            meta['minibatch'] = minibatch
            (xvs, yvs) = self.encode_minibatch(minibatch)

            return (xvs, yvs, meta)

        for it in range(num_iter):
            for ni in range(len(exs) / self.batchsize + 1):
                (xvs, yvs, meta) = sample_minibatch()
                yvs = np.array(yvs, dtype=theano.config.floatX)

                error = self.bprop(*(sum(xvs, []) + [yvs]))
                if ni % 100 == 0:
                    print 'epoch', it, 'iter', ni, '/', len(exs), 'error', error

    def encode_minibatch(self, minibatch):
        '''
        encode minibatch of context + query into vector representation.
        '''
        xvs = []
        yvs = []
        for ex in minibatch:
            context = sum(ex['context'], [])
            context = filter(self.preprocess(context), self.vocab)
            query = filter(self.preprocess(ex['query']), self.vocab)
            blank = query.index('xxxxx')
            context_vector = np.array([self.vocab[word] for word in context], dtype=np.int64)
            query_vectors = []
            cind = 0
            candidates = ex['candidate']
            for (ci, candidate) in enumerate(candidates):
                query[blank] = candidate
                query_vector = np.array([self.vocab[word] for word in filter(self.preprocess(query), self.vocab)], dtype=np.int64)
                query_vectors.append(query_vector)
                if candidate == ex['answer']:
                    cind = ci

            yv = np.zeros(10, dtype=floatX)
            yv[cind] = 1.

            xvs.append([context_vector] + query_vectors)
            yvs.append(yv)
        return (xvs, yvs)


    def test(self, exs):
        all_preds = []
        all_truths = []
        for offset in range(0, len(exs), self.batchsize):
            minibatch = exs[offset:offset + self.batchsize]
            while len(minibatch) < self.batchsize:
                minibatch.append(minibatch[-1])
            (xvs, yvs) = self.encode_minibatch(minibatch)
            yvs = np.array(yvs).astype(theano.config.floatX)
            probs = np.argmax(self.fprop(*(sum(xvs, [])))[0], axis=1)
            truths = np.argmax(yvs, axis=1)
            all_preds.extend(probs[:min(self.batchsize, len(exs)-offset)])
            all_truths.extend(truths[:min(self.batchsize, len(exs)-offset)])
        acc = accuracy(all_preds, all_truths)
        errs = disagree(all_preds, all_truths)
        return (acc, errs)


class LSTMEncoder(BowEmbedLearner):
    def __init__(self, batchsize=1, hidden_dim=100, lr=1e-4):
        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr

        #self.preprocess = lambda sentence:\
        #    remove_stopwords(remove_punctuation(lower(sentence)))

        self.preprocess = lambda sentence: lower(sentence) # less aggressive.


    def compile(self, train_exs):
        exs = train_exs
        self.create_vocab(exs)

        # build LSTM encoder.
        xs = []
        probs = []
        ys = T.matrix('ys')
        params = []

        embed = Embed(self.vocab_size, self.hidden_dim)
        lstm = LSTM(self.hidden_dim)
        params.extend(embed.params)
        params.extend(lstm.params)

        for bi in range(self.batchsize):
            c = T.lvector('context_' + str(bi))
            c_emb = lstm(embed(c))
            qs = []
            scores = []
            for can_id in range(10):
                q = T.lvector('query_' + str(bi) + '_' + str(can_id))
                q_emb = lstm(embed(q))
                score = dot(c_emb, q_emb)
                qs.append(q)
                scores.append(score)
            score_vector = stack(*scores)
            prob = softmax(score_vector)

            xs.append([c] + qs)
            probs.append(prob)

        probs = stack(*probs)
        loss = -mean(log(probs) * ys)

        print '[compiling forward prop]'
        self.fprop = theano.function(inputs=sum(xs, []), outputs=probs)

        updates = optimizers.Adam(loss, params, alpha=FX(self.lr))
        print '[compiling backward prop]'
        self.bprop = theano.function(inputs=sum(xs, []) + [ys],
                                        outputs=loss, updates=updates, profile=False)
        # self.bprop.profile.print_summary()


class MemoryNetwork(BowEmbedLearner):
    def __init__(self, batchsize=1, hidden_dim=100, hop=1, lr=1e-4):
        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hop = hop

        #self.preprocess = lambda sentence:\
        #    remove_stopwords(remove_punctuation(lower(sentence)))

        self.preprocess = lambda sentence: lower(sentence) # less aggressive.


    def compile(self, train_exs):
        mem_size = 20

        exs = train_exs
        self.create_vocab(exs)

        # build LSTM encoder.
        xs = []
        probs = []
        ys = T.matrix('ys')
        params = []

        question_embed = Embed(self.vocab_size, self.hidden_dim)
        mem_layers = []
        for hi in range(self.hop):
            mem_layer = MemoryLayer(mem_size, self.vocab_size, self.hidden_dim)
            params.extend(mem_layer.params)
            mem_layers.append(mem_layer)
        W = theano.shared(value=(npr.rand(hidden_dim, vocab_size) * 0.01).astype(theano.config.floatX),
                            name='W')
        params.append(W)

        inputs = []

        for bi in range(self.batchsize):
            # input layers.
            xs = []
            for x_id in range(mem_size):
                x = T.lvector('context_' + str(bi) + '_' + str(x_id))
                xs.append(x)
            q = T.lvector('query_' + str(bi))
            u = mean(question_embed(q))

            # memory layers.
            for hi in range(self.hop):
                mem_layer = mem_layers[hi]
                o = mem_layer(xs, u)
                u = o + u

            # output layers.
            a = softmax(dot(W, u))

            inputs.extend(xs + [q])
            probs.append(a)

        probs = stack(*probs)
        loss = -mean(log(probs) * ys)

        print '[compiling back prop]'
        self.fprop = theano.function(inputs=inputs, outputs=probs)

        updates = optimizers.Adam(loss, params, alpha=self.lr)
        print '[compiling forward prop]'
        self.bprop = theano.function(inputs=inputs + [ys],
                                        outputs=loss, updates=updates)


    def encode_minibatch(self, minibatch):
        '''
        encode minibatch of context + query into vector representation.
        '''
        xvs = []
        yvs = []
        for ex in minibatch:
            context = sum(ex['context'], [])
            context = filter(self.preprocess(context), self.vocab)
            query = filter(self.preprocess(ex['query']), self.vocab)
            blank = query.index('xxxxx')
            context_vector = np.array([self.vocab[word] for word in context], dtype=np.int64)
            query_vectors = []
            cind = 0
            candidates = ex['candidate']
            for (ci, candidate) in enumerate(candidates):
                query[blank] = candidate
                query_vector = np.array([self.vocab[word] for word in filter(self.preprocess(query), self.vocab)], dtype=np.int64)
                query_vectors.append(query_vector)
                if candidate == ex['answer']:
                    cind = ci

            yv = np.zeros(10, dtype=floatX)
            yv[cind] = 1.

            xvs.append([context_vector] + query_vectors)
            yvs.append(yv)
        return (xvs, yvs)


    def test(self, exs):
        all_preds = []
        all_truths = []
        for offset in range(0, len(exs), self.batchsize):
            minibatch = exs[offset:offset + self.batchsize]
            while len(minibatch) < self.batchsize:
                minibatch.append(minibatch[-1])
            (xvs, yvs) = self.encode_minibatch(minibatch)
            yvs = np.array(yvs, dtype=theano.config.floatX)
            probs = np.argmax(self.fprop(*(sum(xvs, [])))[0], axis=1)
            truths = np.argmax(yvs, axis=1)
            all_preds.extend(probs[:min(self.batchsize, len(exs)-offset)])
            all_truths.extend(truths[:min(self.batchsize, len(exs)-offset)])
        acc = accuracy(all_preds, all_truths)
        errs = disagree(all_preds, all_truths)
        print 'accuracy', acc
        return (acc, errs)






