# Install

It may be necessary to run some extra steps, see:

* https://github.com/pytorch/fairseq/issues/1199

Specifically:

```
export CUDA_HOME=/cm/shared/apps/cuda10.1/toolkit/10.1.105
module load gcc
python setup.py build_ext --inplace
pip install --editable .
```

# Data

The distillation dataset is downloaded here:

/exp/nandrews/nmt_datasets/distill

Note it must still be binarized using `prepare.sh` in this directory.

Regarding the creation of distillation datasets:

* https://github.com/pytorch/fairseq/issues/1220

In particular:

"Hi @MultiPath. In the provided distillation datasets, I found that
valid.en-de was also distilled. Is it necessary that valid data need
to be distilled? Thank you!"

"Hi, I think you can try training a bit longer (400K updates instead
of 300K), and perform checkpoint average over the last five
checkpoints. We usually found that average the last checkpoints get
better performance. The best model I can get from this argument is
around 26.9~27.1 on test set."

"@raymondhs No, we didn't use compound_split_bleu.sh."

# Training