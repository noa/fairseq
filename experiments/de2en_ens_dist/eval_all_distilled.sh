#!/usr/bin/env bash

set -u

ckpt=best
JOB_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en_ens_distill

for dir in ${JOB_DIR}/*/
do
    job=`basename ${dir}`
    bleu=`./eval_distilled.sh ${job} ${ckpt} 2> /dev/null | grep ^BLEU`
    echo "${job} ${bleu}"
done

#eof
