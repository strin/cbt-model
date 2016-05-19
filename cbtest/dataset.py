# utils for CBT dataset.
import re
import pickle
from pprint import pprint
from cbtest.config import cbt_cn_test

def read_cbt(path, limit=None):
    with open(path, 'r') as f:
        exs = []
        context = []
        for line in f:
            line = line.replace('\n', '')
            if line == '':
                continue
            m = re.match(r'[0-9]* ', line).end()
            line_no = int(line[:m-1])
            sentence = line[m:]
            if line_no == 21: # process query.
                sentence = sentence.split('\t')
                query = sentence[0].strip().split(' ')
                answer = sentence[1].strip()
                candidate = sentence[3].strip().split('|')
                candidate = [c for c in candidate if c]
                while len(candidate) < 10:
                    candidate.append('<null>')
                assert(len(candidate) == 10)
                ex = {
                    'context': context,
                    'query': query,
                    'answer': answer,
                    'candidate': candidate
                }
                assert(len(context) == 20)
                exs.append(ex)
                if limit and len(exs) > limit:
                    break
                context = []
            else:
                context.append(sentence.strip().split(' '))
        return exs


def copy_cbt(exs):
    '''
    make a deep copy of cbt dataset
    '''
    return pickle.loads(pickle.dumps(exs))


def lower(words):
    return [word.strip().lower() for word in words]


def filter(words, vocab):
    return [word for word in words if word in vocab]


def unkify(words, vocab, unk='<unk>'):
    return map(lambda word: word if word in vocab else unk, words)


def remove_punctuation(words):
    return [word for word in words if re.match(r'[a-zA-Z\-]+', word)] # TODO: avoid removing things like *bird's*

en_stopwords = None
def remove_stopwords(words):
    from nltk.corpus import stopwords
    global en_stopwords
    if not en_stopwords:
        en_stopwords = set(stopwords.words('english'))
    return [word for word in words if word not in en_stopwords]


