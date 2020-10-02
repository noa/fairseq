#!/usr/bin/env bash

set -u

ckpt=10
split=test
JOB_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en_ens_distill

# for dir in ${JOB_DIR}/*/
# do
#     job=`basename ${dir}`
#     bleu=`./eval_distilled.sh ${job} ${ckpt} 2> /dev/null | grep ^BLEU`
#     echo "${job} ${bleu}"
# done

for dir in ${JOB_DIR}/*FROM_SCRATCH*/
do
    job=`basename ${dir}`
    bleu=`./eval_distilled.sh ${job} ${ckpt} 2> /dev/null | grep ^BLEU`
    calib=`./calibration.sh ${split} ${job}/checkpoint_avg.pt`
    args=`echo "${calib}" | grep Namespace`
    top1=`echo "${calib}" | grep "Top 1 ECE:"`
    top5=`echo "${calib}" | grep "Top 5 ECE:"`
    echo "${job}\n${args}\n${bleu} ${top1} ${top5}"
done

#eof
