#!/usr/bin/env bash

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/de2en

EVAL_DIR=EVAL_RESULTS
mkdir -p ${EVAL_DIR}

for model in m1 m2 m3 m4 m5 m6 m7; do
    if [ ! -f ${EVAL_DIR}/${model}_eval.txt ]; then
	echo "Evaluating ${model}"
	./eval.sh ${model} > ${EVAL_DIR}/${model}_eval.txt
    fi
done

for model in m1xe m2xe m3xe m4xe m5xe m6xe m7xe; do
    if [ ! -f ${EVAL_DIR}/${model}_eval.txt ]; then
	echo "Evaluating ${model}"
	./eval.sh ${model} > ${EVAL_DIR}/${model}_eval.txt
    fi
done

#eof
