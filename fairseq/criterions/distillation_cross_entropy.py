# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def smooth(lprobs, temperature):
    """ lprobs (N x C): N distributions over C classes (log probs)
        temperature (float): temperature to smooth the distributions

    As temperature -> +infinity, the distribution approaches argmax.
    As temperature -> 0, the distribution approaches uniform over C.

    """

    if temperature == 1.0:
      return lprobs

    lprobs = lprobs * temperature  # p^T
    return lprobs - F.logsumexp(lprobs)  # renormalize


def smooth_partial(lprobs, inds, temperature):
    """lprobs (N x K): N sets of K log probabilities
        inds (N x K): N sets indices corresponding to the C events of `lprobs`
        temperature (float): temperature to smooth the distributions

    This is a version of temperature annealing that "does the right
    thing" when the distribution being smoothed is partially specified
    by K probabilities (K < C). We first tally the mass covered by
    `lprobs` and add a dummy event to the distribution to accounting
    for the remaining probability mass. This ensures that
    re-normalization accounts for the unspecified events.

    If the temperature is 1, this has no effect.

    As temperature -> +infinity, the distribution approaches argmax.

    As temperature -> 0, the distribution approaches the uniform.

    """

    if temperature == 1.0:
        return lprobs

    log_mass = torch.logsumexp(lprobs, -1)
    log_remainder = torch.log(1. - torch.exp(log_mass))
    dist_lprobs = torch.cat([log_remainder.reshape(-1, 1), lprobs], -1)
    soft_lprobs = smooth(dist_lprobs, temperature)
    return soft_lprobs[:, 1:]


def partial_kl_div(teacher_lprobs, student_lprobs, inds):
    """Evaluate KL on a subset of events specified by `inds`.

    KL[ teacher, student ] =

      \sum_{i in inds} teacher[i] * (log(teacher[i]) - log(student[i]))

    """
    log_p = teacher_lprobs
    p = torch.exp(log_p)
    log_q = student_lprobs.gather(1, inds)
    kl = p * (log_p - log_q)
    return kl.sum(-1)


@register_criterion('distillation_cross_entropy')
class DistillationCrossEntropyCriterion(FairseqCriterion):
    """ [1] Distilling the Knowledge in a Neural Network
            Hinton et al.
            https://arxiv.org/abs/1503.02531

    """

    def __init__(self, task, sentence_avg, distill_temperature=1.0,
                 teacher_weight=0.1, teacher_top_k=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.temperature = distill_temperature
        assert self.temperature > 1e-5
        self.teacher_weight = teacher_weight
        assert teacher_weight >= 0 and teacher_weight < 1
        self.K = teacher_top_k
        print(f"Teacher top K: {self.K}")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--distill-temperature', default=1.0, type=float,
                            help='Temperature for distillation')
        parser.add_argument('--teacher-weight', default=0.1, type=float,
                            help='Relative weight of the teacher relative to the NLL')
        parser.add_argument('--teacher-top-k', default=32, type=int,
                            help='Top K probabilities in the teacher predictive dist')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def get_normalized_probs(self, model, sample, reduce=False):
        net_output = model(**sample['net_input'])
        return model.get_normalized_probs(net_output, log_probs=True)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        print(f'lprobs shape: {lprobs.shape}')
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        # Calculate the distillation loss
        soft_lprobs = smooth(lprobs, self.temperature)
        vals = sample['teacher_vals'].reshape(-1, self.K)
        inds = sample['teacher_inds'].reshape(-1, self.K)
        soft_targets = smooth_partial(vals, inds, self.temperature)

        # Compute KL[teacher(T), student(T)]
        distill_loss = partial_kl_div(soft_targets.detach(),
                                      soft_lprobs,
                                      inds)

        # Account for padding
        pad_mask = target.eq(self.padding_idx).reshape(-1)
        distill_loss.masked_fill_(pad_mask, 0.)
        if reduce:
            distill_loss = distill_loss.sum()

        # According to [1]: "Since the magnitudes of the gradients
        # produced by the soft targets scale as 1/(T^2) it is
        # important to multiply them by T^2 when using both hard and
        # soft targets. This ensures that the relative contributions
        # of the hard and soft targets remain roughly unchanged if the
        # temperature used for distillation is changed while
        # experimenting with meta-parameters."
        distill_loss_scaled = distill_loss * self.temperature ** 2

        # According to [1]: "We found that the best results were
        # generally obtained by using a condiderably lower weight on
        # the second objective function."
        loss = (1. - self.teacher_weight) * nll_loss + self.teacher_weight * distill_loss_scaled

        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
