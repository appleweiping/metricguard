# Statistical comparison

MetricGuard compares two prediction files case by case. The sign convention is
always `candidate - baseline`. The gate additionally records an optimization
direction and orients this raw delta so positive means improvement. All built-ins
are higher-is-better; lower-is-better plugins require `--direction lower`.

## Alignment contract

The two files must contain exactly the same case IDs. For each ID, `reference` and
the tag set must agree, although record order, tag order, and `prediction` may differ. This catches
common comparison errors such as evaluating a candidate on a revised reference set
or accidentally dropping hard cases.

If an undefined policy skips a case, both reports must skip it. MetricGuard rejects
a one-sided skip instead of silently comparing different populations.

## Bootstrap semantics

The paired bootstrap draws case deltas with replacement and computes their macro
mean. The reported interval is a two-sided percentile interval using linear (R-7)
quantiles. `probability_improvement` is descriptive bootstrap mass above zero in
the configured direction, with ties given half weight. It is not a posterior
probability that the candidate is better.

The two-sided p-value uses a separate paired sign-flip Monte Carlo randomization
test. Under its sharp null, baseline and candidate labels are exchangeable within
each case, so each delta sign is flipped deterministically from a versioned SHA-256
stream. An add-one correction prevents zero Monte Carlo p-values. The validity of
this test depends on paired exchangeability; the percentile interval and p-value
answer related but distinct questions. `--samples` controls the number of replicates
for both procedures, so the smallest possible reported Monte Carlo p-value is
`1 / (samples + 1)`.

Resampling is deterministic from a versioned SHA-256 index stream. This makes an
audit rerunnable across supported Python versions without depending on the private
state format of `random.Random`. Changing the seed, sample count, confidence, case
scores, or MetricGuard algorithm version may change the interval.

## Gates

`--minimum-delta` gates the direction-oriented observed improvement. Its default
is zero, which fails a direct regression. `--minimum-lower-bound` is optional and
gates the oriented lower confidence bound; set it to zero only when CI should
require evidence that the candidate is non-regressing.

The bootstrap measures uncertainty from resampling the supplied cases. It cannot
detect a biased benchmark, data contamination, invalid references, or a metric that
does not represent product quality.
