def accuracy(preds, truths):
    agrees = [float(pred == truth) for (pred, truth) in zip(preds, truths)]
    return sum(agrees) / len(preds)
