# BiCVM
For our published baseline we implemented a version of [BiCVM](https://arxiv.org/pdf/1312.6173.pdf) that treats every sentence as a composition of the word vectors that comprise the sentence.
## Installation instructions
### Important note
Our implementation of BiCVM takes a while to run the more sentences their are in the corpus. For 1,000,000 parallel sentences, expect a runtime of around 16 minutes per iteration. To replicate our results, run with the settings shown in this document. 
### Training for the first time (no previous model to load from)
`python3 bicvm.py --vec 40 --sen 1000000 --itr 12 --lrt 0.2 --neg 15 --mar 40`

### Loading a model and training it
`python3 bicvm.py --vec 40 --sen 1000000 --itr 12 --lrt 0.2 --neg 15 --mar 40 --ten bicvm_theta --idx bicvm_lookup`

Note: don't change the vector size from what the previous model was using as I'm sure it'll crash.

## Comparison to our simple baseline
Our simple baseline achieved the following score.

`Precision: 1.00,  Recall: 0.183, FScore: 0.309`

BiCVM achieved the following score when trained as described above.

`Precision: 0.926, Recall: 0.256, FScore: 0.401`