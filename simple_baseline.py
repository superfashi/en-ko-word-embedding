import gzip
import pickle
import os
from operator import itemgetter
from math import floor
from collections import defaultdict
from functools import partial

assoc = defaultdict(partial(defaultdict, int))

def simple_association(eng, kor):
    l_eng = len(eng)
    l_kor = len(kor)
    if l_eng >= l_kor:
        ratio = floor(l_eng / l_kor)
        for i in range(l_kor):
            if i == l_kor - 1:
                assoc[' '.join(eng[i * ratio :])][kor[i]] += 1
            else:
                assoc[' '.join(eng[i * ratio : (i + 1) * ratio])][kor[i]] += 1
    else:
        ratio = floor(l_kor / l_eng)
        for i in range(l_eng):
            if i == l_eng - 1:
                assoc[eng[i]][' '.join(kor[i * ratio :])] += 1
            else:
                assoc[eng[i]][' '.join(kor[i * ratio : (i + 1) * ratio])] += 1


if __name__ == '__main__':
    if os.path.exists('cache/baseline'):
        with open('cache/baseline', 'rb') as f:
            assoc = pickle.load(f)
    else:
        with gzip.open('train.txt.gz', 'rt', encoding='utf_8') as f:
            for l in f:
                eng, kor = l.split('\t')
                simple_association(eng.split(), kor.split())
        with open('cache/baseline', 'wb') as f:
            pickle.dump(assoc, f)

    with open('val.txt', 'rt', encoding='utf_8') as f:
        with open('val_pred.txt', 'wt', encoding='utf_8') as fw:
            for l in f:
                eng, kor, _ = l.strip().split('\t')
                if eng in assoc:
                    s = list(map(itemgetter(0), sorted(assoc[eng].items(), key=itemgetter(1), reverse=True)))
                    print(kor in s[:5], file=fw)
                else:
                    print(False, file=fw)