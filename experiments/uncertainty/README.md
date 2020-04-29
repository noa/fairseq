Additional reference translations:

```
/exp/nandrews/nmt_datasets/uncertainty/analyzing-uncertainty-nmt
```

SacreBLEU can be used to measure BLEU between multiple outputs and
references:

```
cat output.detok.txt | sacrebleu REF1 [REF2 ...]
```

Note that the system output and references will all be tokenized
internally.

A pretrained model is available here:

```
/exp/nandrews/fairseq/models/ott2018/wmt16.en-de.joined-dict.transformer
```

The `fairseq-generate` script can be used to sample translations from
the model. Some relevant flags:

* `--sampling`: sample hypotheses instead of using beam search
* `--iter-decode-with-external-reranker`: if set, the last checkpoint are assumed to be a reranker to rescore the translations
* `--beam`: beam size

The `fairseq-generate` command produces confidences (in log base 2, I believe):

"This generation script produces three types of outputs: a line
prefixed with O is a copy of the original source sentence; H is the
hypothesis along with an average log-likelihood; and P is the
positional score per token position, including the end-of-sentence
marker which is omitted from the text."

See: https://fairseq.readthedocs.io/en/latest/getting_started.html#evaluating-pre-trained-models