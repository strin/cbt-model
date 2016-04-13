from cbtest.baseline.nonlearning import sliding_window, run_nonlearning
from cbtest.dataset import *
from cbtest.config import *

task = os.environ['task']
test_path = globals()['cbt_' + task + '_test']

exs = read_cbt(cbt_cn_test)
pprint(exs[:1])
run_nonlearning(sliding_window, exs)
