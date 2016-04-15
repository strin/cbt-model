# basic embeding models as baselines.
from cbtest.common import *
from cbtest.layers import (log, dot, mean, softmax, Embed, floatX)
from cbtest.utils import choice
from cbtest.dataset import remove_stopwords, lower, remove_punctuation

import theano.tensor as T

def train_embed_bow(exs,
                    batchsize=128,
                    hidden_dim=100,
                    lr=1e-3,
                    num_iter=100):
    # first pass, create vocab.
    vocab = {}
    def add_word_if_not_exist(word):
        if word not in vocab:
            vocab[word] = len(vocab)

    for ex in exs:
        for sen in ex['context']:
            for word in sen:
                add_word_if_not_exist(word)
        for word in ex['query']:
            add_word_if_not_exist(word)
        for word in ex['candidate']:
            add_word_if_not_exist(word)

    vocab_size = len(vocab)
    print '[vocab size]', vocab_size
    print '[num train exs]', len(exs)
    print '[batchsize]', batchsize

    # build neural net.
    xs = []
    probs = []
    ys = T.matirx('ys')
    params = []

    embed = Embed(vocab_size, hidden_dim)
    params.append(embed.get_params().values())

    for bi in range(batchsize):
        c = T.lvector('context_' + str(bi))
        c_emb = mean(embed(c))
        qs = []
        scores = []
        for can_id in range(10):
            q = T.lvector('query_' + str(bi) + '_' + str(can_id))
            q_emb = mean(embed(q))
            score = dot(c_emb, q_emb)
            qs.append(q)
            scores.append(score)
        score_vector = stack(*scores)
        prob = softmax(score_vector)

        xs.append([c] + qs)
        probs.append(prob)

    probs = stack(*probs)
    loss = -mean(log(probs) * ys)

    updates = optimizers.Adam(loss, params, alpha=lr)
    bprop = theano.function(inputs=sum(xs, []) + [ys],
                                    outputs=loss, updates=updates)

    # second pass, learn embedding.
    def sample_minibatch():
        minibatch = choice(exs, size=batchsize, replace=True)
        xvs = []
        yvs = []
        for ex in minibatch:
            context = sum(ex['context'], [])
            context = remove_stopwords(remove_punctuation(lower(context)))
            query = remove_stopwords(remove_punctuation(lower(ex['query'])))
            blank = query.index('xxxxx')
            context_vector = np.array([vocab[word] for word in context], dtype=np.int64)
            query_vectors = []
            cind = 0
            for (ci, candidate) in enumerate(ex['candidate']):
                query[blank] = candidate
                query_vector = np.array([vocab[word] for word in query], dtype=np.int64)
                query_vectors.append(query_vector)
                if candidate == ex['answer']:
                    cind = ci

            yv = np.zeros(10, dtype=floatX)
            yv[cind] = 1.

            xvs.append([context_vector] + query_vectors)
            yvs.append(y)

        return (xvs, yvs)

    for it in range(num_iter):
        (xvs, yvs) = sample_minibatch()
        yvs = np.array(yvs)

        error = bprop(*sum(xvs, []), yvs)
        print 'iter', it, 'error', error









