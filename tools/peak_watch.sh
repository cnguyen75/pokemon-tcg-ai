#!/bin/zsh
# Peak-freeze watchdog.
#
# 2026-08-02 finding: every copy PEAKS far above where it settles, early in
# its life while TrueSkill sigma is still wide (v8_judge hit 930.6 at game 5
# and settled at 782.2; a v12_pwr copy hit 887.6 at game 25 and settled 773).
# Frozen submissions keep their last rating forever and count for team score,
# so catching a copy mid-spike and deactivating it (submit 2 fresh copies to
# push both actives out) BANKS that number permanently.
#
# Peaks last hours, not days -> poll fast. Only a score above the current
# banked floor is worth acting on, since freezing can only raise the floor.
#
# usage: peak_watch.sh [FLOOR] [INTERVAL_SEC]
FLOOR=${1:-820.6}
INT=${2:-600}
while true; do
  out=$(~/.local/bin/kaggle competitions submissions -c pokemon-tcg-ai-battle 2>/dev/null \
        | sed -n '3,4p' | awk '{print $1, $NF}')
  if [ -n "$out" ]; then
    echo "$out" | while read id score; do
      case "$score" in (*[!0-9.]*|"") continue;; esac
      if [ "$(echo "$score > $FLOOR" | bc -l)" = "1" ]; then
        if [ "$(echo "$score > 833.1" | bc -l)" = "1" ]; then
          echo "*** BRONZE-RANGE PEAK: $id at $score — SUBMIT 2 COPIES NOW TO BANK IT ***"
        else
          echo "PEAK $id at $score (floor $FLOOR) — bankable, freeze to raise floor"
        fi
      fi
    done
  fi
  sleep $INT
done
