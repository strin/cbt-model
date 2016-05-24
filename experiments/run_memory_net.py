from cbtest.dataset import *
from cbtest.utils import *
from cbtest.config import *
from cbtest.baseline.embedding import *
from cbtest.evaluate import Experiment
import random
import pdb, traceback, sys

import argparse

parser = argparse.ArgumentParser(description='LSTM Embedding Baseline')
parser.add_argument('--task', type=str, default='cn')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--iter', type=int, default=20)
parser.add_argument('--encoder', type=str, default='bow')
parser.add_argument('--memory', type=str, default='lexical')
parser.add_argument('--small', type=bool, default=False)
parser.add_argument('--PE', type=str, default=True) # Position Encoding.

args = parser.parse_args()
print colorize('[arguments]\t' + str(args), 'red')

task = args.task
print '[running task]', task
train_path = globals()['cbt_' + task + '_train']
test_path = globals()['cbt_' + task + '_test']
print '[train_path]', train_path
print '[test_path]', test_path

try:
    if args.small:
        train_exs = read_cbt(train_path, limit=1000)
    else:
        train_exs = read_cbt(train_path)
    test_exs = read_cbt(test_path)

    learner = CBTLearner(batchsize=64, hidden_dim=100, lr=args.lr, position_encoding=args.PE)
    learner.create_vocab(train_exs)
    learner.preprocess_dataset(train_exs)
    learner.preprocess_dataset(test_exs)

    if args.memory == 'lexical':
        learner.mem_size = 1024
        learner.unit_size = 1
        learner.sen_maxlen = 128 # query sentence len.
        learner.encode_context = learner.encode_context_lexical
        learner.encode_query = learner.encode_query_lexical
        learner.arch = learner.arch_memnet_lexical
    elif args.memory == 'window':
        param_b = 2
        learner.mem_size = 1024
        learner.unit_size = 2 * param_b + 1
        learner.sen_maxlen = 2 * param_b + 1
        learner.encode_context = learner.encode_context_window
        learner.encode_query = learner.encode_query_window
        learner.arch = learner.arch_memnet_lexical
    elif args.memory == 'sentence':
        learner.mem_size = 20
        learner.unit_size = 1024
        learner.sen_maxlen = 1024
        learner.encode_context = learner.encode_context_sentence
        learner.encode_query = learner.encode_query_sentence
        learner.arch = learner.arch_memnet_lexical

    learner.compile()

    experiment = Experiment('memory-net-%s' % args.encoder)

    for it in range(args.iter):
        learner.train(train_exs, num_iter=1)
        (acc, errs) = learner.test(test_exs)
        print '[epoch %d]' % it, 'accuracy = ', acc
        experiment.log_json(result={
            'task': task,
            'acc': acc,
            'errs': errs
        })
        experiment.next()
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

print 'saving model...'
experiment.log_pickle(fprop=learner.fprop,
        bprop=learner.bprop)
