from cbtest.dataset import *
from cbtest.config import *
from cbtest.baseline.embedding import train_embed_bow

task = os.environ.get('task')
if not task:
    task = 'cn'
print '[running task]', task

train_path = globals()['cbt_' + task + '_train']
test_path = globals()['cbt_' + task + '_test']

print '[train_path]', train_path
print '[test_path]', test_path

train_exs = read_cbt(test_path)

train_embed_bow(train_exs)


