import random

VAL_SIZE = 500
TEST_SIZE = 500

def pick_good_pair():
    kor, eng = word_pairs.pop()
    if ', ' in eng:
        eng = eng.split(', ')[0]
    return (eng, kor)

def pick_bad_pair():
    eng = random.choice(eng_words)
    if ', ' in eng:
        eng = eng.split(', ')[0]
    kor = random.choice(kor_words)
    return (eng, kor) if dic[kor] != eng else pick_bad_pair()

def generate_eval(name):
    with open(name, 'wt', encoding='utf_8') as f:
        for _ in range(VAL_SIZE):
            if random.random() < 0.5:
                print(*pick_good_pair(), "True", sep='\t', file=f)
            else:
                print(*pick_bad_pair(), "False", sep='\t', file=f)

if __name__ == '__main__':
    dic = dict()
    with open('dataset/kengdic_2011.tsv', encoding='utf_8') as f:
        for l in f:
            s = l.split('\t')
            if s[3] != "NULL":
                dic[s[1]] = s[3]
    word_pairs = list(dic.items())
    random.shuffle(word_pairs)
    kor_words, eng_words = list(dic.keys()), list(dic.values())
    generate_eval('val.txt')