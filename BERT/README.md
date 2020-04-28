# BERT
## How it works
We use the [transformers module](https://github.com/huggingface/transformers) to load a pertained multilingual BERT. 

Then for each word in our validation dataset, we encode the word and obtain the 768 dimensional vector corresponding to the [CLS] tag. 

We write these vectors to a file corresponding to the language the words are in. We create magnitude files from these vectors and query the top 5 most similar Korean words to the given English word vector. 

## Performance
Without any fine-tuning, mBERT performed as follows on our validation dataset.

`Precision: 0.886 Recall: 0.285 F1: 0.43077`