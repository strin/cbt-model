# utils for CBT dataset.
import re
from pprint import pprint
from cbtest.config import cbt_cn_test

def read_cbt(path):
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
                ex = {
                    'context': context,
                    'query': query,
                    'answer': answer,
                    'candidate': candidate
                }
                assert(len(context) == 20)
                exs.append(ex)
                context = []
            else:
                context.append(sentence.strip().split(' '))
        return exs


def lower(words):
    return [word.strip().lower() for word in words]


def remove_punctuation(words):
    return [word for word in words if re.match(r'[a-zA-Z\-]+', word)] # TODO: avoid removing things like *bird's*


def remove_stopwords(words):
    from nltk.corpus import stopwords
    return [word for word in words if word not in set(stopwords.words('english'))]


