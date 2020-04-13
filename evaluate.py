from sklearn.metrics import classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gold', type=str, required=True)
parser.add_argument('--pred', type=str, required=True)

def read_gold(file):
    labels = []
    with open(file, 'rt', encoding='utf_8') as f:
        for line in f:
            labels.append(line.strip().split('\t')[2])
    return labels

def read_pred(file):
    labels = []
    with open(file, 'rt', encoding='utf_8') as f:
        for line in f:
            labels.append(line.strip())
    return labels

def display_prf(true, pred):
    res = classification_report(true, pred, digits=5)
    print(res)

if __name__ == '__main__':
    args = parser.parse_args()
    gold = read_gold(args.gold)
    pred = read_pred(args.pred)
    display_prf(gold, pred)