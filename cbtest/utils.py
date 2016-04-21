from cbtest.common import *

def normalize_log(arr):
    arr = np.array(arr)
    max_ele = np.max(arr)
    arr -= max_ele
    arr -= np.log(np.sum(np.exp(arr)))
    return arr

def choice(objs, size, replace=True, p=None):
    all_inds = range(len(objs))
    inds = npr.choice(all_inds, size=size, replace=replace, p=p)
    return [objs[ind] for ind in inds]

def TV(logprob1, logprob2):
    return np.sum(np.abs(np.exp(logprob1) - np.exp(logprob2)))

def mkdir_if_not_exist(path):
    if path == '':
        return
    if not os.path.exists(path):
        os.makedirs(path)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
