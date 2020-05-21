#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import os
import numpy as np
import h5py
from scipy.special import logsumexp
import torch

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
from fairseq.options import add_distributed_training_args

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.validate')


def get_ensemble_lprobs(task, sample, models, criterion):
    lprobs = []
    n_models = len(models)
    for model in models:
      lp = task.predict_step(sample, model, criterion).cpu().numpy()
      if n_models < 2:
        return lp
      lprobs.append(lp)
    lprobs = np.stack(lprobs)
    return logsumexp(lprobs, axis=0, b=float(n_models))


def main(args, override_args=None):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    print(f"Torch see CUDA? {torch.cuda.is_available()}")
    print(f"Use CPU? {args.cpu}")
    print(f"Use CUDA? {use_cuda}")

    overrides = {'criterion': 'cross_entropy'}

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    if args.print_full_dist:
        assert args.full_dist_path
        import pickle
        if os.path.exists(args.full_dist_path):
            print(f"Deleting existing file: {args.full_dist_path}")
            os.remove(args.full_dist_path)

        if args.storage_format == 'pickle':
          dist_output_file = open(args.full_dist_path, 'ab')
        elif args.storage_format == 'hdf5':
          dist_output_file = h5py.File(args.full_dist_path, 'w', libver='latest')
        else:
          raise ValueError(args.storage_format)

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            print("Using CUDA")
            model.cuda()

    # Print args
    logger.info(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    for subset in args.valid_subset.split(','):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception('Cannot find dataset: ' + subset)

        # How big is the dataset?
        n_examples = len(dataset)
        print(f"Number of examples: {n_examples}")

        if args.storage_format == 'hdf5' and args.print_full_dist:
          # Create the datasets
          lprobs_dataset = dist_output_file.create_dataset("lprobs", (n_examples,),
                                                           dtype=h5py.vlen_dtype('float32'))
          indices_dataset = dist_output_file.create_dataset("indices", (n_examples,),
                                                            dtype=h5py.vlen_dtype('int32'))

        # Initialize data iterator
        print(f"Num workers: {args.num_workers}")
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if args.print_full_dist:
                target = sample['target']
                #lprobs = task.predict_step(sample, model, criterion)
                lprobs = get_ensemble_lprobs(task, sample, models, criterion)
                vals, inds = lprobs.topk(args.dist_top_k)
                ids = sample['id'].cpu().numpy()
                keep = np.logical_not(target.eq(criterion.padding_idx).cpu().numpy())
                vals = vals.cpu().numpy()
                inds = inds.cpu().numpy()
                n_sentences = ids.shape[0]
                for j in range(n_sentences):
                    kj = keep[j]
                    if args.storage_format == 'pickle':
                      pickle.dump((ids[j], vals[j][kj], inds[j][kj]),
                                  dist_output_file)
                    elif args.storage_format == 'hdf5':
                      id_ = ids[j]
                      lprobs_dataset[id_] = vals[j][kj].flatten()
                      indices_dataset[id_] = inds[j][kj].flatten()
                    else:
                      raise ValueError(args.storage_format)
            elif args.measure_calibration:
                target = sample['target']
                lprobs = task.predict_step(sample, model, criterion).cpu().numpy()
                keep = np.logical_not(target.eq(criterion.padding_idx).cpu().numpy())
                log_outputs.append({
                  'lprobs': lprobs[keep],
                  'targets': target.cpu().numpy()[keep]})
            else:
                _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
                progress.log(log_output, step=i)
                log_outputs.append(log_output)

        if args.print_full_dist:
            if args.storage_format == 'pickle':
                dist_output_file.close()
            elif args.storage_format == 'hdf5':
                dist_output_file.close()
            else:
                raise ValueError(args.storage_format)
        elif args.measure_calibration:
            # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/calibration.py
            import tensorflow as tf
            from tensorflow_probability.python.stats.calibration import brier_score
            from tensorflow_probability.python.stats.calibration import expected_calibration_error
            labels = np.concatenate([batch['targets'] for batch in log_outputs], 0)
            logits = np.concatenate([batch['lprobs'] for batch in log_outputs], 0)
            with tf.device("/cpu:0"):
              print("Computing Brier score...")
              print("TODO: Should be computed incrementally to be more memory efficient.")
              b = tf.reduce_mean(brier_score(labels, logits))
              print(f"Brier score: {b}")
              print("Computing ECE...")
              ece = expected_calibration_error(10, logits=logits,
                                               labels_true=labels)
              print(f"ECE: {ece}")
        else:
            with metrics.aggregate() as agg:
                task.reduce_metrics(log_outputs, criterion)
                log_output = agg.get_smoothed_values()

            progress.print(log_output, tag=subset, step=i)


def cli_main():
    parser = options.get_validation_parser()
    add_distributed_training_args(parser)
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    add_distributed_training_args(override_parser)
    override_args = options.parse_args_and_arch(override_parser,
                                                suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == '__main__':
    cli_main()
