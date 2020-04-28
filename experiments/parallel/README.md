The description of steps to reproduce the parallel model experiments
is here:

https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation

The data is downloaded here:

/exp/nandrews/fairseq/parallel

Regarding generating new distillation datasets:

https://github.com/pytorch/fairseq/issues/1554

* "It should be the same as running generate with a pretrained autoregressive model by passing `--gen-subset train`."

And some suggested commands:

```
srun --gres gpu:1 fairseq-generate     data-bin/wmt16_en_de_bpe32k     --path checkpoint.avg20.pt  --beam 4 --lenpen 0.6 --gen-subset train  > distill_txt/distill_full_0.txt


python examples/backtranslation/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 3 --output extract_txt/distill_full_0 --srclang en --tgtlang de distill_txt/distill_full_0.txt
```

However, it seems like there may be some issues still:

* https://github.com/pytorch/fairseq/issues/2003

Additional references / discussion:

* https://github.com/pytorch/fairseq/issues/1463
* https://github.com/pytorch/fairseq/issues/1600
* https://github.com/pytorch/fairseq/issues/1736