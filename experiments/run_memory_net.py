from cbtest.dataset import *
from cbtest.utils import *
from cbtest.config import *
from cbtest.baseline.embedding import *
from cbtest.evaluate import Experiment
import random

import argparse

parser = argparse.ArgumentParser(description='LSTM Embedding Baseline')
parser.add_argument('--task', type=str, default='cn')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--iter', type=int, default=20)
parser.add_argument('--encoder', type=str, default='bow')

args = parser.parse_args()
print colorize('[arguments]\t' + str(args), 'red')

task = args.task
print '[running task]', task
train_path = globals()['cbt_' + task + '_train']
test_path = globals()['cbt_' + task + '_test']
print '[train_path]', train_path
print '[test_path]', test_path

#train_exs = read_cbt(train_path, limit=1000)
train_exs = read_cbt(train_path)
test_exs = read_cbt(test_path)

learner = MemoryNetwork(batchsize=32, hop=3, hidden_dim=args.dim,
                        lr=args.lr, encoder=args.encoder,
                        flags={
                            'position_encoding': False
                        })
learner.compile(train_exs)

experiment = Experiment('memory-net-%s' % args.encoder)

for it in range(args.iter):
    learner.train(train_exs, num_iter=1)
    (acc, errs) = learner.test(test_exs)
    print '[epoch %d]' % it, 'accuracy = ', acc
    experiment.log(result={
        'task': task,
        'acc': acc,
        'errs': errs
    })




