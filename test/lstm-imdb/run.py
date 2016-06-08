from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle
from cbtest.utils import ns_add, ns_merge, Namespace
from cbtest.layers import (log, dot, mean, softmax, Embed, floatX, stack, LSTM, MemoryLayer, LinearLayer, position_encoding, LSTMq)
import cbtest.optimizers as optimizers

import gzip
import os

import numpy
import theano
import theano.tensor as T
import time


def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((n_samples, maxlen)).astype('int64')
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s

    return x, labels


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    return dataset


def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    path = get_dataset_file(
        path, "imdb.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pickle.load(f)
    test_set = pickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for (s, t) in zip(params, tparams):
        t.set_value(s)


def unzip(tparams):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = []
    for t in tparams:
        new_params.append(t.get_value())
    return new_params


def build_model(ns):
    x = T.matrix('x', dtype='int64')
    y = T.vector('y', dtype='int64')
    n_timesteps = x.shape[1]
    n_samples = x.shape[0]
    params = []

    embed_layer = Embed(ns.n_words, ns.dim_proj)
    params.extend(embed_layer.params)
    v0 = embed_layer(x.flatten()).reshape((n_samples, n_timesteps, ns.dim_proj))

    lstm = LSTM(ns.dim_proj)
    params.extend(lstm.params)
    v1 = lstm(v0.dimshuffle(1, 0, 2))

    # TODO: average over only non-zero elements. will it help?
    proj = mean(v1, axis=0)
    linear = LinearLayer(ns.dim_proj, ns.ydim)
    params.extend(linear.params)
    prob = softmax(linear(proj))

    fprop = theano.function([x], prob, name='fprop')
    pred = theano.function([x], prob.argmax(axis=1), name='fpred')

    off = floatX(1e-8) # avoid nan.
    if prob.dtype == 'float16':
        off = floatX(1e-6)
    loss = -log(prob[T.arange(n_samples), y] + off).mean()

    updates = optimizers.Adam(loss, params, alpha=floatX(ns.lrate))
    bprop = theano.function(inputs=[x, y],
                            outputs=loss,
                            updates=updates,
                            on_unused_input='ignore')

    if 'decay_c' in ns and ns.decay_c > 0.:
        print('using decay_c')
        decay_c = theano.shared(floatX(ns.decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (linear.W ** 2).sum()
        weight_decay *= decay_c
        loss += weight_decay

    return Namespace(fprop=fprop, pred=pred, bprop=bprop, x=x, y=y, loss=loss, params=params)


def pred_error(pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = pred(x)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - floatX(valid_err) / len(data[0])

    return valid_err


def train_lstm(ns):
    global load_data
    global prepare_data

    print('Loading data')
    train, valid, test = load_data(n_words=ns.n_words, valid_portion=0.05,
                                   maxlen=ns.maxlen)

    if ns.test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:ns.test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    assert(min(min(test[0])) >= 1) # 0 is not used.

    ydim = numpy.max(train[1]) + 1

    print('Building model')

    m = build_model(ns_add(ns, ydim=ydim))

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), ns.valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), ns.valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if ns.validFreq == -1:
        ns.validFreq = len(train[0]) // ns.batch_size
    if ns.saveFreq == -1:
        ns.saveFreq = len(train[0]) // ns.batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(ns.max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), ns.batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = m.bprop(x, y)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, ns.dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if ns.saveto and numpy.mod(uidx, ns.saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(m.params)
                    numpy.savez(ns.saveto, *params, history_errs=history_errs)
                    pickle.dump(ns, open('%s.pkl' % ns.saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, ns.validFreq) == 0:
                    train_err = pred_error(m.pred, prepare_data, train, kf)
                    valid_err = pred_error(m.pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(m.pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(m.params)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )

                    if (len(history_errs) > ns.patience and
                        valid_err >= numpy.array(history_errs)[:-ns.patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > ns.patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, m.params)
    else:
        best_p = unzip(m.params)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), ns.batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if ns.saveto:
        numpy.savez(ns.saveto, *best_p, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


if __name__ == '__main__':
    param = Namespace(
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=100,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        #optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset='imdb',

        # Parameter for extra option
        test_size=500,  # If >0, we keep only this number of test example.
    )

    train_lstm(param)
