# Jaicodyn: possession from automatic tracking

Team Jaicodyn, More than a Hack 2027 possession challenge. Code: https://github.com/jyoon926/jaicodyn

## 1. Approach

We label every frame of BAR-ATH as home possession (H), away possession (A) or dead ball (D) with
a **three-state hidden Markov decoder** (Viterbi), built on hand-designed per-frame evidence from
the tracking and the automatic smart tagging.

**Why a decoder and not a per-frame classifier.** The challenge scores possession *spells*
(spellF1), not frames, and asks for *controlled* possession: a touch or deflection does not
change hands. Both are statements about time, not single frames. In a Viterbi decoder they become
one parameter each: a switch between states costs a fixed amount of evidence. Every score is in
"seconds of evidence", so `turnover = 0.5 s` reads as "the other team must be clearly nearest the
ball for about half a second before possession changes". When there is no evidence (a contested
or loose ball, or the ball unseen), the path keeps its current state, which is exactly the rule
"possession stays with the last team that had it". A frame classifier would need all of this
added as post-processing, and would still flicker in duels.

**Why hand-designed evidence and not a learned model.** The provided game is the only one with
this camera and tracker, and it has no labels we may use. The labelled games come from other
providers with different tracking quality: idsse sees the ball on every frame, BAR-ATH loses it
on 41% of frames. A model fitted to clean data learns weights for conditions the test match
doesn't have. Each evidence term is a physical statement (a player within 1.5 m, a ball still for
a second, players walking), and each constant was set from its meaning, then tuned on labelled
games made to look like BAR-ATH (section 3).

### Evidence per frame

| Term | What it says | Strength |
|---|---|---|
| **Nearest team** | The team clearly nearer the ball has it: `tanh((d_other − d_team) / 1 m)` | up to ±0.5 / s |
| **Gate** | Applied to the term above. Full when someone is within 1.5 m of the ball, fading to nothing at 2.5 m; zero if the ball is unseen or out of play | × 0–1 |
| **Live** | A player at the ball is weak evidence the ball is in play | 1.0 / s × gate |
| **Touch** | The ball changed velocity by ≥ 3 m/s (over 0.2 s either side) while one team was clearly nearest. That team gets evidence for the 0.4 s around the touch | 0.5 / s |
| **Dead-ball cues** | Ball beyond the lines (more than 1 m, and not a brief exit that comes straight back); ball still for a second; players walking relative to the match's own median speed; ball unseen for over 10 s; a set-piece tag starting. Ball moving at pass speed counts against | ±0.5–2 / s each |
| **Set-piece team** | Smart-tagged corners, goal kicks and free kicks name the taking team (checked label-free: 18 of 18 corners and goal kicks agree with the attacking direction). Added as P = 0.95 for that team over the first 3 s after the decoder's restart | log-odds |
| **Phase tags** | Inside ball gaps only: smart-tagging phases (build-up, progression, transitions…) name a team in or out of possession, weighted by an estimated reliability of 0.6–0.95 per code, at half strength | ≤ 0.5 / s |

Switch costs: H↔A 0.5 s, H/A→D 1.0 s, D→H/A 1.0 s. Each half is decoded separately.

## 2. Assets used and dropped

**Used**
- **Tactical tracking** (`BAR-ATH_tactical_atd`): ball and player positions. Players are
  de-duplicated per team (tracks within 1 m merged), because the tracker often draws two boxes on
  one player. The ball is cleaned of teleports (single-frame jumps faster than 40 m/s), and holes
  up to 1 s are bridged linearly.
- **Smart tagging** (`BAR-ATH_tactical_smart_tagging.xml`): set-piece tags as dead-ball cues and
  as the restart team, and phase tags as team evidence inside ball gaps. These are Metrica's
  automatic output, not human annotation. None of their settings could be checked against labels,
  so each comes from a label-free check on BAR-ATH itself (`jaicodyn/decode/tags.py`).

**Not used**: the video, the homography (both only in our debugging viewer), and the broadcast
feed.

**Tried and dropped**, each measured on degraded training games:

| Idea | Result | Why dropped |
|---|---|---|
| Gap classifier: a gradient-boosted model of H/A/D inside ball gaps, trained on the decoder's own confident frames | No gain | Inside real gaps only dead-ball frames were confident, so it learned "unseen means dead" |
| Restart-team model: which team restarts after a stoppage | +0.012 on train, +0.004 on validation (one game worse) | Within noise; a perfect restart call is worth only about +0.05 |
| Down-weighting evidence just after restarts | +0.003 | Noise |
| "Ghost" players: keep a team present when the tracker merges its player into an opponent | Merges cost almost nothing (0.542 with, 0.541 without); the ghost lost 0.001–0.006 | No problem to fix |
| Deterministic anchor rules: lock possession where one team is sustainedly in control, decode between | Not impactful | Left in the code, off |
| Flight rule: a fast ball (> 8 m/s) with no touch near gives no nearest-team evidence | +0.008 on train; lost 11 of 16 held-out games | Did not carry over |
| Strong touch evidence (weight ≥ 2) | −0.03 | An opponent's single touch isn't controlled possession |

Details of the restart work are in [docs/restarts.md](docs/restarts.md).

## 3. Training data

No label from BAR-ATH was used for anything. The 5-minute public window was used once per
version, to check that frames and labels line up, never to choose a setting.

| Dataset | Source and licence | Labels and who made them | Matches used |
|---|---|---|---|
| **idsse** (Bundesliga) | Bassek, Rein, Weber and Memmert, *Scientific Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y. CC BY 4.0, approved for this challenge | DFL's official per-frame `BallPossession` and `BallStatus` flags, from the league's data provider | 7 (5 train, 2 validation) |
| **SkillCorner** (A-League) | github.com/SkillCorner/opendata (MIT, credit requested) | The phases-of-play files published with SkillCorner's open data (team in possession per phase) | 20 (14 train, 3 validation, 3 test) |
| **Metrica** (anonymised) | github.com/metrica-sports/sample-data (credit requested) | We derived per-frame labels from Metrica's event logs: the last on-ball event's team holds until a stoppage event | 3 (test) |

All 30 are full matches. No model is trained on them: they were used only to choose the
decoder's constants and to check them.

**Labels to the challenge's definition.** idsse's possession flag changes on every touch (raw
spells average 5–9 s, against 10–15 s once relabelled). We relabel it to *controlled*
possession: a team spell under 3 s that interrupts the other team's possession is folded back
into the possession it interrupted, unless it comes straight after a restart. The 3 s comes from
the challenge text ("a player keeps the ball for a few seconds"), not from any score. SkillCorner
phases and Metrica's event-derived labels already require control and are used as they are.

**Making the labelled games look like BAR-ATH.** Clean idsse would tune the decoder for conditions
the test match doesn't have, so we degrade it, measuring everything from BAR-ATH's own tracking
and tags without its labels (`scripts/degrade.py`, `config/degrade.json`):

- **Missing ball**: hidden on 40.8% of frames, in gaps drawn from BAR-ATH's own gaps of 5 s or
  more. Dead frames are hidden 2.8× as often as live ones; that ratio is estimated from player
  speeds and checked against the set-piece tags. Gaps start more often near the lines.
- **Merges**: players next to an opponent vanish into them at BAR-ATH's rate (0.035 per second
  within 1.5 m) and for BAR-ATH's durations.
- **Ball position error**: BAR-ATH's ball track is no rougher than idsse's, but at dribbling
  speed the ball sits 0.3–0.4 m further from the nearest player. We add a smooth offset, 0.55 m
  per axis, drifting over about 8 s, fitted to match those distances without adding roughness.

The same process is applied to Metrica (`metrica_degraded`). SkillCorner is left alone: its
broadcast camera already loses players and ball in its own way.

**Selection protocol.** Constants were tuned on train (degraded idsse and SkillCorner). For the
final choice (touch weight and flight rule), validation and test were pooled and scored per
source. The decision was made on degraded idsse, the closest match to BAR-ATH, and the other
sources could only veto a setting that clearly broke them. As a result **no untouched held-out
score remains**; the scores below are what the choices were made on.

| Source | Matches | spellF1 (strict labels) |
|---|---|---|
| idsse, degraded, train | 5 | 0.533 |
| SkillCorner, train | 14 | 0.538 |
| idsse, degraded, validation | 2 | 0.529 |
| Metrica, degraded, test | 3 | 0.475 |
| SkillCorner, validation + test | 6 | 0.492 |
| Metrica, clean, test | 3 | 0.569 |
| idsse, clean, validation | 2 | 0.643 |

## 4. Possession rules implemented

- **Controlled possession.** A change of team needs about half a second of the other team being
  clearly nearer the ball (the 0.5 s switch cost). A single touch, a deflection or a ball passing
  an opponent's feet doesn't flip possession. Touches add weak evidence for the touching team: a
  touch by the team already in possession confirms it, and a run of touches by the other team
  helps place the turnover.
- **Loose and contested balls.** When nobody is within 2.5 m of the ball, or both teams are about
  equally close (a duel), there is no team evidence, and possession stays with the last team.
- **Ball unseen.** The state holds through the gap. Inside it, only the dead-ball cues (players
  walking, a long gap, a set-piece tag) and the phase tags can move it.
- **Dead ball.** D is entered when the ball is beyond a line (beyond 1 m, for more than 1 s or
  without coming straight back), is still for a second, players walk together, or a set piece is
  tagged. A player standing beside a ball out of play is taking a restart, not possessing it, so
  the nearest-team evidence is switched off there.
- **Restarts.** Every restart leaves D for a team (it costs 1 s of evidence). For tagged
  corners, goal kicks and free kicks, the tagged team is strongly preferred over the first seconds
  after the restart. Throw-ins and kickoffs carry no team in the tags, so for them the nearest
  player decides. A short spell straight after a restart is kept, since a restart has a clear
  possessor.

## 5. Limitations

- **The missing ball is the main error.** 41% of BAR-ATH's ball frames are missing, mostly in gaps
  over 5 s. Inside a gap the decoder mostly holds its last state, so turnovers and stoppages that
  happen out of sight are found late or missed. On degraded idsse this costs about 0.11 spellF1
  against clean idsse (0.53 vs 0.64).
- **Dead ball on BAR-ATH may be weaker than our proxies show.** On the public window, dead-ball F1
  is 0.58 against 0.82 for each team. We did not tune on this, but it suggests the dead-ball cues,
  set on degraded idsse, carry over imperfectly to this camera.
- **The proxies are imperfect.** Degraded Metrica keeps its own pattern of blanking the ball on
  almost every dead frame. SkillCorner's broadcast camera differs from the wide shot. Part of
  BAR-ATH's extra ball-to-player distance may be playing style rather than tracking error; without
  labels the two can't be separated.
- **Smart-tag reliability is unverified.** The labelled games have no tags. The phase-tag
  reliabilities are upper bounds: Metrica builds the tags from the same tracking, so its errors
  and ours are shared.
- **No held-out score is left**, because test was used for the final choice.

**With more time** we would: detect the ball ourselves in the video inside the gaps, the largest
remaining error; learn the evidence weights jointly on many degraded games rather than tuning a
few constants by grid; and build a better restart-team call for throw-ins and kickoffs from where
the ball left the pitch.

## 6. Reproduction

Needs the challenge data bundle, Python 3.11 and git; about two minutes. Only the competition
match is needed: all settings are written into the code, nothing is trained at run time, and
nothing is random.

```bash
git clone --recurse-submodules https://github.com/jyoon926/jaicodyn.git
cd jaicodyn
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/MoreThanAHackData        # the folder that holds tactical/
python -m scripts.prepare_data --sources competition
python -m scripts.decode BAR-ATH                       # writes jaicodyn.csv
shasum -a 256 jaicodyn.csv
# d908ab74ce41f9610c12ea21c4cd3aa583d35f81cf8292b98b16916ea5556bf1
```

To rerun the experiments (downloads the labelled games, about 4.6 GB), see "Full setup" in the
[README](README.md).
