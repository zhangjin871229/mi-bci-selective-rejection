# Deviations from protocol_2b.json (sha256 53a30bc8…, frozen 2026-10-06 23:46:22)

## Deviation 1 — D-block size (2026-10-07, before any E label was read)

**Found:** while building the training caches. The protocol assumed 120 + 120 + 160 = 400 cue trials in the
training sessions, i.e. 360 D trials split into 5 blocks of 72. Three subjects deviate from the data set description:
B04 and B05 have 140 trials in 02T (total 420) and B08 has 160 trials in 01T (total 440); see the T_sessions field
in logs/run_2b.out.

**Change:** the C block is unchanged (last 40 cue trials of 03T). D = all cue trials before the C block, in
recording order and before artifact exclusion, cut into 5 contiguous blocks of as-equal-as-possible size
(numpy.array_split). For 400-trial subjects this is identical to the frozen rule (5 × 72).

**E labels read at that time:** no. No E-label file had been opened by any script; the evaluation script had not
been run. No model output had been inspected.
