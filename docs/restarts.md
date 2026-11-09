# Error analysis and the restart experiments

This section covers where the decoder's remaining error lies, and three things we built to fix the
largest share of it: restart-team evidence in the decoder, a model that predicts which team
restarts, and a change to the evidence just after restarts. We kept the first and dropped the
other two. The results below show why.

## Setup shared by every experiment

- **Data.** The idsse train games (5 Bundesliga matches) with the ball hidden to match the
  competition match's tracking (`idsse_degraded`). That means 40.8% of frames missing, gap lengths
  drawn from the competition match, the ball 2.8 times likelier to be missing while dead than while
  live, and gaps starting more often near the lines. All of these were measured from the
  competition match's tracking and automatic tags, not its labels (`scripts/degrade.py`,
  `config/degrade.json`). The 14 SkillCorner train games are shown alongside. Their broadcast
  camera makes them a weaker proxy, but they add sample size.
- **Labels.** "Strict" labels follow the challenge's definition of possession. A team spell shorter
  than 3 s that interrupts the other team's possession is a touch without control, and is folded
  back into the possession it interrupted. A short spell straight after a restart is kept
  (`fold_uncontrolled` in `jaicodyn/data/labels.py`).
- **Metric.** `spellF1` from the official scorer (`challenge/evaluation/possession_metrics.py`),
  scored against strict labels.
- **Protocol.** Every choice is made on the train split. Validation (2 idsse games, degraded the
  same way) is read once, against a bar fixed in advance: the mean must not drop, and no game may
  drop by more than 0.01. The test split and the competition match's public window are never used
  to choose anything.
- **Baseline.** The decoder as submitted scores `spellF1` 0.496 on degraded idsse train and 0.478
  on SkillCorner train.

## 1. Where the error is: oracle tests

To find out which errors are worth fixing, we used the true labels to fix one kind of error
perfectly, leaving everything else as the decoder had it, and measured how far `spellF1` rose.
All fixes run within a half. Oracles use labels, so they ran on the train split only.

### Dead-ball errors (`scripts/oracle_dead.py`)

The decoder's team calls are kept. The true labels decide only where dead ball is.

| Fix, applied alone | Degraded idsse (5) | SkillCorner (14) |
|---|---|---|
| Snap the end of each predicted stoppage to the true end | +0.023 (5/5 games) | +0.026 (14/14) |
| Delete predicted stoppages that overlap no true one | +0.019 (5/5) | +0.071 (14/14) |
| Insert true stoppages no prediction overlaps | +0.008 (5/5) | +0.015 (13/14) |
| Dead ball exactly right (ceiling) | +0.051 | +0.121 |

On degraded idsse, the decoder predicts 49.6% dead against 46.2% true. Recall is 0.96 and
precision 0.89. Three quarters (74%) of the extra dead frames lie at the edges of real stoppages.
Predicted stoppages start on time (median 0 s) but run on a median 0.8 s past the restart. The
rest are 57 stoppages, across the 5 games, that did not happen.

### Team errors (`scripts/oracle_team.py`)

The decoder's dead ball is kept. The true labels correct only who has the ball. A turnover is a
change from H to A, or A to H, between live frames. Predicted and true turnovers are matched one
to one, same direction, within 5 s.

| Fix, applied alone | Degraded idsse (5) | SkillCorner (14) |
|---|---|---|
| Correct the team after each restart, up to the first true turnover | **+0.098** (5/5) | +0.043 (14/14) |
| Insert missed turnovers | **+0.096** (5/5) | +0.051 (14/14) |
| Move matched turnovers to the true frame | +0.052 (5/5) | +0.056 (14/14) |
| Delete false turnovers | +0.040 (4/5) | +0.036 (14/14) |
| All four | +0.311 | +0.192 |
| Teams exactly right wherever both say live (ceiling) | +0.335 | +0.213 |

Per game on degraded idsse, the decoder finds 71 turnovers (a median 1.1 s from the true time)
and invents 47, and it misses 66. After **34% of restarts** (172 of 502) it gives the ball to the
wrong team. On SkillCorner that figure is 21% (335 of 1,591). Team errors are a far larger pool
than dead-ball errors, and the restart team is the largest single fix. So that's what we went
after.

The four fixes overlap, so each figure is the most that fix could gain on its own, not what it
adds on top of the others.

## 2. Restart-team evidence in the decoder (kept)

The decoder takes an optional P(home restarts) for each restart (`restart_p_home` in
`jaicodyn/decode/__init__.py`). It isn't used in the submission, because no source of that
probability earned its place (sections 3 and 4).

### First design: a cost per stoppage (replaced)

Our first design made the cost of each restart depend on how likely that team was to take it:

    cost(D -> X) = base + scale × (−log p_X),   base = restart_s − scale × ln 2

We chose this base so that an uninformative p = 0.5 costs exactly what a restart cost before,
`restart_s`. That gave a clean test of the wiring. Feeding p = 0.5 everywhere changed **0 labels
on 20 games (1,697,516 frames) at three values of the scale**, and the per-frame probabilities
were unchanged to the last digit.

A unit test with a confident p then exposed a flaw. The decoder restarted to the favoured team,
then handed the ball to the other team one frame later. Restarting cheaply and turning the ball
over straight away cost 0.3 + 0.5 = 0.8 s of evidence. Paying for the restart it believed in
would have cost 7.2 s. Viterbi only looks one step back, so it can't see that those two switches
are one move. A 1-frame possession is close to the worst output for `spellF1`.

### Final design: evidence over the first seconds after each restart

The probability now becomes evidence on the first `restart_window_s` (3 s) of each restart
possession. That's `restart_scale × log(p_H / p_A) / 2` per second for home, and the same amount
against away. Over a whole window the gap between the two teams is `restart_scale × log(p_H /
p_A)` seconds of evidence, the same as the cost design, but the decoder collects it only by
keeping that team in possession. A first pass without restart evidence places the restarts; the
second pass decodes with the evidence.

- **The p = 0.5 test passes the same way:** 0 labels differ on the same 20 games and three scales,
  and the largest posterior difference is 0.0. At p = 0.5 the evidence is exactly 0.
- **The one-frame flip is gone.** In the case that broke the cost design, the restart team now
  keeps the ball for the rest of the 4 s clip.

### What a perfect restart call is worth through this design

We fed in the true restart team, with P = 0.999 for the team the true labels next give the ball
to:

| | Degraded idsse (5) | SkillCorner (14) |
|---|---|---|
| Restart team wrong, decoder alone | 34% | 21% |
| True team, 3 s window | **+0.048** (5/5), 0% wrong | **+0.020** (14/14), 0.1% wrong |
| True team, 6 s window | +0.041 (5/5) | +0.003 (8/14) |
| Oracle ceiling for the restart team (section 1) | +0.098 | +0.043 |

Scaling the evidence anywhere from 1 to 8 gives the same result, because at p = 0.999 it already
decides the restart. So the design captures about half of the oracle's gain. Of the 158 restarts
it corrected on degraded idsse, 56% were later handed to the other team rather than ending in a
stoppage. 41% ended within 1 s of the window closing. Restart possessions were a median 5.1 s
long, against 9.7 s in the true labels, and 47% ended more than 2 s early.

## 3. Weaker evidence just after restarts (dropped)

**Hypothesis.** Just after a restart, the player nearest the ball is often an opponent closing
the taker down. If so, that evidence would hand the ball back once the window closes. Weakening it
for the first T seconds should let the restart team hold until a real turnover.

**Test.** Keep a share ρ of the nearest-player evidence for T seconds after each restart, with the
true restart team fed in as above:

| With the true restart team | Degraded idsse (5) | SkillCorner (14) |
|---|---|---|
| Restart evidence alone (3 s window) | +0.048 | +0.020 |
| Plus T = 5 s, ρ = 25–50% | +0.046 to +0.047 | +0.019 to +0.020 |
| Plus T = 8 s, ρ = 50% (best) | +0.051 | +0.021 |
| Plus T = 12 s, ρ = 50% | +0.051 | +0.017 |
| Plus T = 8–12 s, ρ = 0% | +0.027 to −0.002 | −0.017 to −0.059 |
| T = 8 s, ρ = 25%, no restart evidence | +0.001 | −0.009 |

**Result.** The best setting gains +0.003 over restart evidence alone, which is noise. Removing
the evidence altogether for longer hurts, badly on SkillCorner, because the same evidence catches
real turnovers soon after restarts. On its own it does nothing. The hypothesis was wrong. The
remaining half of the oracle's gain doesn't come from misleading evidence after restarts.
Correcting the restart team "up to the first true turnover" also removed some of the decoder's
own turnover errors, which no restart call can reach. So about +0.05 is the realistic ceiling for
restart-team work through the decoder. We removed this setting.

## 4. A model of which team restarts (dropped)

### Features

A restart is a switch from dead to live in the first-pass decode. Every feature is positive for
home and negative for away, and changes sign if the teams are swapped. The logistic regression
has no intercept, so the model can't learn a home bias from a few hundred restarts. The direction
each team attacks comes from the team centroids, since goalkeepers pull a team's centroid towards
its own goal.

| Idea | Feature |
|---|---|
| Out-of-play rule | The team nearest the ball just before it left the pitch, split by touchline, goal line, or still on the pitch (a foul). The other team should restart. Also which team had possession before the stoppage. |
| Restart location gives the type | Where the ball is when play resumes, otherwise where it went out. **Corner**: the team attacking that end. **Goal area**: goal kick, the team defending it. **Centre spot after a goal-line exit**: kickoff by the team that conceded. |
| Formation | Each team's player nearest the restart spot (the taker stands at the ball). Players in the penalty box at that end, weighted by who attacks it. Both come from player positions, so they work while the ball is unseen. |
| Kickoff rules | The conceding team kicks off after a goal, as above. Opening kickoffs carry no information about the team and get none. |

### Training and checks

The model was trained on the 512 restarts in the degraded idsse train games, with the true team
taken from strict labels. To check that it generalises across games, each game was predicted by a
model trained on the other four.

**A timing bug we found and fixed.** In the first version, the "last touch" features were read up
to 1 s after the stoppage began. They pointed to the team that went on to restart 60–72% of the
time, the opposite of the out-of-play rule, because they were catching the player walking to the
ball. We changed them to the 0.3–1.5 s before the ball is first seen beyond a line.

**The out-of-play rule can't be read from this tracking.** With the ball hidden around stoppages
as in the competition match, the ball is seen beyond a touchline before only 4% of restarts, and
beyond a goal line before only 3%. When it is, the feature is at chance (47% and 50%). The
kickoff-after-goal feature never fired. Goals are rare, and the ball must be seen at the centre
spot.

How often each feature fired, and how often it pointed to the team that restarted (first
version):

| Feature | Fires on | Points to the restarting team |
|---|---|---|
| Taker (nearest to the restart spot) | 92% | 77% |
| Corner | 2% | 78% |
| Goal kick | 12% | 79% |
| Possession before the stoppage | 100% | 55% |
| Penalty-box crowd | 75% | 54% |
| Kickoff after a goal | 0% | — |

Fitted coefficients (final version): taker +1.58, corner +1.02, penalty-box crowd +0.57, last
touch at a goal line −0.56, last touch on the pitch +0.52, possession before +0.38, possession
before (on the pitch) −0.31, goal kick +0.17 (largely the same information as the taker, since the
goalkeeper is both), last touch at a touchline +0.14, kickoff after a goal 0.00.

### Results

| | Restart team right | `spellF1` |
|---|---|---|
| Train, each game predicted by a model fitted on the others (5) | 65% → 76% | 0.496 → 0.508 (+0.012, 4/5 games) |
| Validation (read once): J03WMX | 70% → 65% | 0.525 → 0.522 (−0.003) |
| Validation (read once): J03WR9 | 60% → 70% | 0.516 → 0.528 (+0.012) |
| Validation mean | 65% → 67% | +0.004 |

**Decision.** The model passed the bar we had fixed. But the train gain mostly failed to carry
over, and one validation game got worse. A +0.004 gain over two games is within noise, and the
model would still have to transfer from idsse to the competition's camera. Given section 2's
ceiling of about +0.05 for a perfect call, a 70%-accurate model is worth about +0.01. We dropped
it.

## 5. Smart-tagging set pieces as the restart team (kept)

The competition's automatic tags name the team for corners, goal kicks and free kicks. Throw-ins
and kickoffs are tagged `N/A`. The labelled games have no tags, so this could not be validated
against labels. Every setting comes from a label-free check on the competition match itself
(`jaicodyn/decode/tags.py`).

- **What the tag team means.** Of the corners with the ball visible, every one was tagged with the
  team attacking that end (9 of 9). Of the goal kicks, every one was tagged with the team defending
  it (9 of 9). We read the attacking directions from the team centroids. So the tag names the team
  taking the set piece.
- **Confidence.** That agreement (18 of 18) with Laplace's rule gives P = 19/20 = 0.95.
- **Timing.** On all 27 set pieces with a team, the decoder's restart falls 2–14 s after the tag
  starts. The tag's team therefore covers 1 s before the tag to 15 s after it, and goes through
  the restart evidence of section 2.

Effect on the competition match:

| Set piece | Decoder's first team = tag team, before | After |
|---|---|---|
| Corner kick (12) | 25% | 100% |
| Goal kick (10) | 60% | 90% |
| Free kick (5) | 60% | 100% |

1.03% of graded frames change label. The decoder finds 280 possessions instead of 278, and none
is shorter than 1 s. On the labelled games nothing changes, because they have no tags.

## What we did not try

- **Smart-tagging phase teams as possession evidence.** The phase tags (build-up, progression,
  transitions and so on) carry a team and cover 82% of graded frames. Their team agrees with the
  decoder 83–87% of the time for build-up and progression, and the "defending" codes name the other
  team. We did not try this as possession evidence.
- **Sharper taker cues.** Examples are the thrower standing beyond the touchline, the goalkeeper
  at a goal kick, and the closest approach to the spot over a few seconds rather than one frame.
  All are limited by the same ceiling.

## Reproducing

Commit `0944cd4` still contains the deleted code. The restart model is in
`jaicodyn/decode/restart.py`, and `python -m scripts.restart_model --validate` runs it. The
post-restart setting is `post_restart_s` and `post_restart_who_w` in `Params`. The oracles
(`scripts/oracle_dead.py`, `scripts/oracle_team.py`) and the restart evidence are still in the
current code.

Two of the experiments were one-off checks, not saved scripts: the true restart team fed through
the restart evidence (section 2), and the post-restart sweep (section 3). Both took
`restart_p_home` as 0.999 for the team the true labels next give the ball to and 0.001 otherwise,
and scored with the official scorer against strict labels. Section 3 also used the post-restart
setting from commit `0944cd4`.

Every result here needs the processed idsse and SkillCorner games
(`python -m scripts.prepare_data`) and the degraded idsse games (`python -m scripts.degrade`).
