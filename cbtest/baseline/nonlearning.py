from cbtest.dataset import remove_stopwords, lower, remove_punctuation

def maximum_frequency(ex):
    context = ex['context']
    context = sum(context, []) # concatenate context.
    context = remove_stopwords(remove_punctuation(lower(context)))
    max_score = -float('inf')
    answer = None
    for candidate in ex['candidate']:
        count = 0.
        for word in context:
            if word == candidate:
                count += 1
        score = count
        if score > max_score:
            max_score = score
            answer = candidate
    return answer



def sliding_window(ex):
    context = ex['context']
    context = sum(context, []) # concatenate context.
    context = remove_stopwords(remove_punctuation(lower(context)))
    query = remove_stopwords(remove_punctuation(lower(ex['query'])))
    blank = query.index('xxxxx')
    max_score = -float('inf')
    answer = None
    for candidate in ex['candidate']:
        query[blank] = candidate
        query_len = len(query)
        score = 0.
        for i in range(0, len(context) - query_len):
            window = set(context[i:i + query_len]) # BOW.
            for word in query:
                if word in window:
                    score += 1
        score /= len(context) - query_len
        if score > max_score:
            max_score = score
            answer = candidate
    return answer


def run_nonlearning(method, exs, debug=False):
    acc = 0.
    for ex in exs:
        proposed = method(ex)
        if debug:
            print '[proposed]', proposed, '[answer]', ex['answer']
        if proposed == ex['answer']:
            acc += 1
    return acc / len(exs)

