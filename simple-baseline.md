# Simple Baseline

For each sentence pair read from `train.txt.gz`, uses space character to separate English and Korean sentences into word chucks. Align them roughly by their position in the sentence, for example, if `X Y Z` is translated into `A B C`, then the baseline model will associate `(X, A)`, `(Y, B)` and `(Z, C)`. If the words cannot align perfectly, some words will be append to the last word.

By doing so, we have an association dictionary, which stores all the word pairs that the above steps aligned, along with the raw count of the pairwise occurrences. 

Reading in the validation set, we label the English-Korean pair `True` if the word pair occurs in the top five most counted word-pairs for that English word, `False` otherwise.