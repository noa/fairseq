See: https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt#3-train-a-model

To obtain confidences from trained models, use the `predict.sh` script. This calls
`fairseq_cli/validate.py` with two additional arguments:

* `--write-full-dist`: tell it to write the confidences
* `--full-dist-path` where to write them
* `--dist-top-k` how many of the top predictions to write

For ensembles, because an entire ensemble would not fit on a single
GPU, it recommended to write the confidences for each ensemble
separately, and then manually average them.

The confidences are saved in binary pickle format, with each record
being a Python tuple of the form

`(id, lprobs, inds)`

where:

* `id` is the example ID
* `lprobs` are the top K log probabilities (base e)
* `inds` are the subword indices associated with those values

The `id`s are non-sequential but (as far as I can tell) deterministic.
This implies that confidences from two different models on the same
dataset will be written in the same order, making it easy to combine
them for ensembles.


