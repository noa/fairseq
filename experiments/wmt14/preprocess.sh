#! /usr/bin/env bash

DATA=/expscratch/nandrews/fairseq/wmt14_en_de

# Binarize the dataset
fairseq-preprocess --source-lang en \
		   --target-lang de \
		   --trainpref $DATA/train \
		   --validpref $DATA/valid \
		   --testpref $DATA/test \
		   --destdir $DATA/bin/wmt17_en_de \
		   --thresholdtgt 0 \
		   --thresholdsrc 0 \
		   --workers 20

# eof
