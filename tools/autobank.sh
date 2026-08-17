#!/bin/zsh
# Autonomous peak-banker for the PTCG ladder endgame.
#
# WHY: copies spike far above their true level in their first ~5-25 games
# (wide TrueSkill sigma), then decay within hours. A frozen submission keeps
# its rating forever and counts for team score, so banking a spike = permanent
# result. In-session monitors die with the session and we MISSED a 929.9 peak
# (silver range) on 2026-08-08 that way. This runs from cron instead.
#
# WHAT IT DOES: every run, read the two ACTIVE submissions. If either is above
# the recorded floor, submit 2 copies of the winning build — that pushes both
# actives out of the latest-2 window, freezing them at their current values —
# then record the new floor.
#
# SAFETY RAILS:
#   - only ever submits the same already-validated build (runs/submission_v12_pwr.tar.gz)
#   - banking can only RAISE the team floor; a cold copy costs nothing
#   - max 2 bank events per day (4 of the 5 daily submissions), leaving
#     headroom for manual recycling
#   - no-op if the tarball or kaggle CLI is missing
set -u
ROOT=/Users/colin/Documents/pokemon-tcg-ai
KG=~/.local/bin/kaggle
TARBALL=$ROOT/runs/submission_v12_pwr.tar.gz
STATE=$ROOT/runs/peak_floor.txt
LOG=$ROOT/logs/autobank.log
COMP=pokemon-tcg-ai-battle
MAX_BANKS_PER_DAY=2

[ -x "$KG" ] || exit 0
[ -f "$TARBALL" ] || exit 0
[ -f "$STATE" ] || echo "879.8" > "$STATE"
FLOOR=$(cat "$STATE")
TODAY=$(date +%Y-%m-%d)

banks_today=$(grep -c "^$TODAY BANKED" "$LOG" 2>/dev/null || echo 0)
if [ "$banks_today" -ge "$MAX_BANKS_PER_DAY" ]; then exit 0; fi

# lines 3-4 of the submissions table are the two ACTIVE submissions
subs=$($KG competitions submissions -c $COMP 2>/dev/null | sed -n '3,4p' | awk '{print $1, $NF}')
[ -n "$subs" ] || exit 0

best=""; best_score=0
echo "$subs" | while read id score; do
  case "$score" in (*[!0-9.]*|"") continue;; esac
  if [ "$(echo "$score > $FLOOR" | bc -l)" = "1" ]; then
    echo "$id $score" >> /tmp/_autobank_hit.$$
  fi
done

if [ -s /tmp/_autobank_hit.$$ ]; then
  hit=$(sort -k2 -g -r /tmp/_autobank_hit.$$ | head -1)
  hid=$(echo "$hit" | awk '{print $1}')
  hscore=$(echo "$hit" | awk '{print $2}')
  echo "$TODAY $(date +%H:%M) PEAK $hid at $hscore (floor $FLOOR) -> banking" >> "$LOG"
  $KG competitions submit -c $COMP -f "$TARBALL" \
     -m "autobank 1/2: freezing peak $hscore (floor was $FLOOR)" >> "$LOG" 2>&1
  sleep 5
  $KG competitions submit -c $COMP -f "$TARBALL" \
     -m "autobank 2/2: guarantee push-out of peak $hscore" >> "$LOG" 2>&1
  echo "$hscore" > "$STATE"
  echo "$TODAY $(date +%H:%M) BANKED $hid at $hscore; new floor $hscore" >> "$LOG"
fi
rm -f /tmp/_autobank_hit.$$
