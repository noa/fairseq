NOTE: In this configuration there are about 4000 steps per epoch.

This experiment proceeds as follows:

1. Load an existing checkpoint.
2. Reset the optimizer with a new schedule, cosine decay with restarts with a fixed period.
3. Train for N periods saving a checkpoint after each one.

The N checkpoints may then be used for stochastic weight averaging
(SWA) or to produce an ensemble.

Example usage:

```bash
./qsub.sh e1b checkpoint_last
```

This will use checkpoint `checkpoint_last.pt` from in the existing job
directory `e1b` as the starting point for a new ensemble. It may be
useful to start from earlier checkpoints than the very last one, to
obtain more diversity, in which case you'd just adjust
`checkpoint_last` to `checkpoint51`, for example.

Details:

* The number of cycles is controlled by `--lr-period-updates`
  and `--max-update`. For example, to perform `10` cycles set
  these as `100` and `1000`.

* To save a checkpoint at the end of every cycle, assuming a constant
  cycle length, use `--save-interval-updates N` where `N` is set to
  the same value as `--lr-period-updates`, e.g. `100` above.

* The `--lr` and `--max-lr` parameters control the min and max
  learning rates for each cycle.

* The `--t-mult` parameter controls the length of each cycle.
  To have cycles of constant length, this can be set as `1`.

* The `--lr-shrink` argument can be used to reduce the max learning
  rates for later cycles. To keep the max learning rate constant, this
  can be set to `1`.