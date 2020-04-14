# Data

## train.txt.gz

This is a training dataset that consists of English-Korean sentence aligned pairs. Each line is a pair with a `\t` character as a separator. The dataset is sanitized such that the English sentence does not contain any non-English characters, and Korean sentence does not contain any non-Korean characters (except for numeric numbers). 

Notice that the dataset is very large so we compressed it with `.gz` format.

## val.txt

The validation set we prepared for evaluate the model's performance. Every line consists of three parts: an English word, a Korean word, and a label. If the label is `True`, then that means the English-Korean word-pair semantic relationship, `False` otherwise.

---

The full data can be downloaded here: [https://drive.google.com/drive/folders/1GlcLb14dYMhiBzg_U3qWOVNU54N_SRs2](https://drive.google.com/drive/folders/1GlcLb14dYMhiBzg_U3qWOVNU54N_SRs2)