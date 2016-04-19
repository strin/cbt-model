from cbtest.dataset import *
from cbtest.config import *
from cbtest.baseline.embedding import BowEmbedLearner

task = os.environ.get('task')
if not task:
    task = 'cn'
print '[running task]', task

train_path = globals()['cbt_' + task + '_train']
test_path = globals()['cbt_' + task + '_test']

print '[train_path]', train_path
print '[test_path]', test_path

train_exs = read_cbt(train_path)
test_exs = read_cbt(test_path)

learner = BowEmbedLearner(batchsize=1, hidden_dim=100, lr=1e-4)
learner.compile(train_exs)

for it in range(100):
    learner.train(train_exs, num_iter=1)
    learner.test(test_exs)



