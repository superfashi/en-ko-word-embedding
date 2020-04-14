import gzip
import itertools
import string
from translate.storage.tmx import tmxfile

def is_english(char: str):
    o = ord(char)
    return o == 32 or o == 45 or 48 <= o <= 57 or 65 <= o <= 90 or 97 <= o <= 122 or 192 <= o <= 255

def is_korean(char: str):
    o = ord(char)
    return o == 32 or 48 <= o <= 57 or 0xac00 <= o <= 0xd7a3

def process(eng: str, kor: str):
    eng = ''.join(filter(is_english, eng)).strip()
    kor = ''.join(filter(is_korean, kor)).strip()
    if eng and kor:
        return (eng, kor)
    return None

def read_open_subtitles():
    with gzip.open('dataset/en-ko.tmx.gz') as f:
        tmx_file = tmxfile(f, 'en', 'ko')
    for node in tmx_file.unit_iter():
        yield process(node.getsource(), node.gettarget())

def read_wiki_matrix(threashold: float = 0.):
    with gzip.open('dataset/WikiMatrix.en-ko.tsv.gz', 'rt', encoding='utf_8') as f:
        for line in f:
            split = line.split('\t')
            if float(split[0]) >= threashold:
                yield process(split[1], split[2])

if __name__ == '__main__':
    with gzip.open('train.txt.gz', 'wt', encoding='utf_8') as f:
        for line in itertools.chain(read_open_subtitles(), read_wiki_matrix()):
            if line:
                print(*line, sep='\t', file=f)