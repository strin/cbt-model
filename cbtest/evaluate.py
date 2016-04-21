from datetime import datetime
from os import path
from cbtest.utils import mkdir_if_not_exist
import json

def accuracy(preds, truths):
    agrees = [float(pred == truth) for (pred, truth) in zip(preds, truths)]
    return sum(agrees) / len(preds)


def disagree(preds, truths):
    disagrees = [float(pred != truth) for (pred, truth) in zip(preds, truths)]
    return disagrees


class Experiment(object):
    def __init__(self, name):
        self.runid = name + '-' + datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        self.count = 0
        mkdir_if_not_exist(self.runid)

    def log(self, **kwargs):
        for (key, dat) in kwargs:
            with open(path.join(self.runid, str(self.count), key), 'w') as f:
                json.dump(dat, f)
        self.count += 1




