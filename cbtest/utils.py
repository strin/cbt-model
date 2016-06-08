# ML utils.
from cbtest.common import *
import sys
import time

def choice(objs, size, replace=True, p=None):
    all_inds = range(len(objs))
    inds = npr.choice(all_inds, size=size, replace=replace, p=p)
    return [objs[ind] for ind in inds]


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


class Timer(object):
    def __init__(self, name=None, output=sys.stdout):
        self.name = name
        if output and type(output) == str:
            self.output = open(output, 'w')
        else:
            self.output = output

    def __enter__(self):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Start', 'green')
        self.tstart = time.time()
        self.output.flush()

    def __exit__(self, type, value, traceback):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Elapsed: %s' % (time.time() - self.tstart),
                                      'green')
        self.output.flush()


# namespace management.
from argparse import Namespace

def ns_merge(ns1, ns2):
    '''
    merge two namespacs.
    '''
    dict1 = vars(ns1)
    assert(all([name not in ns1 for name in vars(ns2)]))
    dict1.update(vars(ns2))
    return Namespace(**dict1)


def ns_add(ns1, **kwargs):
    ns2 = Namespace(**kwargs)
    return ns_merge(ns1, ns2)
