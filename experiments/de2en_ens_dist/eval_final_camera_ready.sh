#!/usr/bin/env bash

set -e
set -u

# Evaluate constituent models
JOBS_DIR=/expscratch/nandrews/nmt/fairseq/jobs/de2en

# CE
if false; then
    for model in m1xe m2xe m3xe m4xe m5xe m6xe m7xe; do
	bleu=`./eval.sh ${model} 2> /dev/null | grep ^BLEU`
	calib=`./calibration.sh test ${JOBS_DIR} ${model}/checkpoint_avg.pt`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${model}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi

# CE + LS
if false; then
    for model in m1 m2 m3 m4 m5 m6 m7; do
	bleu=`./eval.sh ${model} 2> /dev/null | grep ^BLEU`
	calib=`./calibration.sh test ${JOBS_DIR} ${model}/checkpoint_avg.pt`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${model}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi

# Evaluate ensembles (CE + checkpoint averaging)

e3="m1xe m2xe m3xe"
e5="m1xe m2xe m3xe m4xe m5xe"
e7="m1xe m2xe m3xe m4xe m5xe m6xe m7xe"

if false; then
    for ens in "$e3" "$e5" "$e7"; do
	bleu=`./eval_ensemble_with_ckpt_avg.sh 10 ${ens} 2> /dev/null | grep ^BLEU`
	calib=`./calibration.sh test ${JOBS_DIR} --checkpoint avg ${ens}`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${ens}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi

# Evaluate ensembles (CE + checkpoint averaging)

e3="m1 m2 m3"
e5="m1 m2 m3 m4 m5"
e7="m1 m2 m3 m4 m5 m6 m7"

if false; then
    for ens in "$e3" "$e5" "$e7"; do
	bleu=`./eval_ensemble_with_ckpt_avg.sh 10 ${ens} 2> /dev/null | grep ^BLEU`
	calib=`./calibration.sh test ${JOBS_DIR} --checkpoint avg ${ens}`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${ens}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi

# Evaluate distilled ensembles

JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/de2en_ens_distill

# Evaluate each distilled ensemble, for ensemble sizes in {3, 5, 7}
e3d="de2en_ens3_64_last_fs_96_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_64_last.h5"
e5d="de2en_ens5_64_last_fs_96_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_m4xe_m5xe_64_last.h5"
e7d="de2en_ens7_64_last_fs_96_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_m4xe_m5xe_m6xe_m7xe_64_last.h5"
if false; then
    for job in ${e3d} ${e5d} ${e7d}; do
	bleu=`./eval_distilled.sh ${job} 10 2> /dev/null | grep ^BLEU`
	calib=`./calibration.sh test ${JOBS_DIR} --checkpoint avg ${job}`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${job}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi


# Vary truncation in {64, 128, 256}

de2en32="de2en32_FROM_SCRATCH_kl_1_32_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_m4xe_m5xe_m6xe_m7xe_32_last.h5"
de2en64="de2en_FROM_SCRATCH_kl_1_64_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_m4xe_m5xe_m6xe_m7xe_64_last.h5"
de2en128="de2en128_FROM_SCRATCH_kl_1_128_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_m4xe_m5xe_m6xe_m7xe_128_last.h5"
de2en256="de2en256_FROM_SCRATCH_kl_1_256_0.5_300000_0.0007_0.1_0.1_m1xe_m2xe_m3xe_m4xe_m5xe_m6xe_m7xe_256_last.h5"

if true; then
    for job in ${de2en32} ${de2en64} ${de2en128} ${de2en256}; do
	bleu=`./eval_distilled.sh ${job} 10 2> /dev/null | grep ^BLEU`
	calib=`./calibration.sh test ${JOBS_DIR} --checkpoint avg ${job}`
	args=`echo "${calib}" | grep Namespace`
	top1=`echo "${calib}" | grep "Top 1 ECE:"`
	top5=`echo "${calib}" | grep "Top 5 ECE:"`
	echo "[${job}]  ${bleu:0:13},  ${top1},  ${top5}"
    done
fi

#eof
