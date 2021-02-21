#!/usr/bin/env bash

set -u

JOB_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en

for dir in ${JOB_DIR}/*/
do
    job=`basename ${dir}`
    bleu=`./eval.sh ${job} 2> /dev/null | grep ^BLEU`
    echo "${job} ${bleu}"
done

#eof
