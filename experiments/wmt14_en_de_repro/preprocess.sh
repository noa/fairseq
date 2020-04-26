#! /usr/bin/env bash

DATA=/expscratch/nandrews/fairseq/wmt14_en_de

# Binarize the dataset
fairseq-preprocess --source-lang en \
		   --target-lang de \
		   --trainpref $DATA/train \
		   --validpref $DATA/valid \
		   --testpref $DATA/test \
		   --destdir $DATA/bin/wmt14_en_de_joined \
		   --thresholdtgt 0 \
		   --thresholdsrc 0 \
		   --joined-dictionary \
		   --workers 16

# eof
