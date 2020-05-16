# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import torch

from fairseq.data import LanguagePairDataset

from .translation import load_langpair_dataset, TranslationTask
from . import register_task


@register_task('translation_with_teacher')
class TranslationWithTeacher(TranslationTask):
    """
    Translate from source language to target language with a teacher

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--teacher-pred', required=True,
                            help='Path to teacher distribution')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

        # Open the teacher predictions file
        self.teacher_pred_file = h5py.File(args.teacher_pred, 'r')
