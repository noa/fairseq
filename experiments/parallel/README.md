The description of steps to reproduce the parallel model experiments
is here:

https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation

The data is downloaded here:

/exp/nandrews/fairseq/parallel

Regarding generating new distillation datasets:

https://github.com/pytorch/fairseq/issues/1554

* "It should be the same as running generate with a pretrained autoregressive model by passing `--gen-subset train`."

Additional references / discussion:

* https://github.com/pytorch/fairseq/issues/1463
* https://github.com/pytorch/fairseq/issues/1600
* https://github.com/pytorch/fairseq/issues/1736