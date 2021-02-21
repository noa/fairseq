#!/usr/bin/env bash

set -e
set -u

# Evaluate distilled ensembles
JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_distill

# Evaluate each distilled ensemble, for ensemble sizes in {3, 5, 7}
e3d="en2de_ens3_64_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_ce1_ce2_ce3_64_last.h5"
e5d="en2de_ens5_64_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_ce1_ce2_ce3_ce4_ce5_64_last.h5"
e7d="en2de_ens7_64_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_ce1_ce2_ce3_ce4_ce5_ce6_ce7_64_last.h5"
if true; then
    for job in ${e3d} ${e5d} ${e7d}; do
	#echo "Evaluating: ${job}"
	#./eval_distilled.sh ${job} 10
	bleu=`./eval_distilled.sh ${job} 10 2> /dev/null | grep ^BLEU`
	#echo ${bleu}
	calib=`./calibration.sh test ${JOBS_DIR} --checkpoint avg ${job}`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${job}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi

#eof
