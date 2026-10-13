"""Draw the README figures from results/results.json.

Usage:
    python tools/make_figures.py      # writes figures/ladder.svg and figures/noise.svg

Hand-written SVG in the site style: card #F6F2E8, ink #231F1B, secondary
#3D372F, rule #B3A992; series blue #1F6FB5 and orange #C0601F; Archivo for
text; direct labels, no legend.
"""
import json
import os
from datetime import date
from xml.sax.saxutils import escape

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD, INK, SECONDARY, RULE = "#F6F2E8", "#231F1B", "#3D372F", "#B3A992"
BLUE, ORANGE, PURPLE = "#1F6FB5", "#C0601F", "#7A4E9E"

# Builds named on the ladder figure, with what changed (from the README and
# each submission's own description).
LABELS = {
    "v1.1": "v1 flat search",
    "v4": "v4 scripted policy",
    "v5": "v5 learned value net",
    "v6": "v6 whole-turn beam search",
    "v10_azval": "v10 AlphaZero value head",
    "v11": "v11 deck.csv fix",
    "v12_pwr": "v12 Powerful Hand fix",
}


# Builds whose dots sit close to others: label placed above at (rating, x
# offset in px) and joined to the dot by a leader line.
STACKED = {"v4": (940, 0), "v5": (915, 0), "v6": (890, 16), "v10_azval": (870, 16)}


def head(W, H, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-labelledby="t d" font-family="Archivo, sans-serif">',
            f'<title id="t">{escape(title)}</title>', f'<desc id="d">{escape(desc)}</desc>',
            f'<rect width="{W}" height="{H}" fill="{CARD}"/>']


def text(x, y, s, size=14, fill=INK, weight="400", anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{escape(s)}</text>')


def ladder_svg(res):
    lad = res["ladder"]
    subs = lad["all_submissions"]
    W, H = 920, 470
    L, R, T, B = 70, 740, 96, 400
    d0, d1 = date(2026, 6, 30), date(2026, 8, 11)
    y0, y1 = 500, 950

    def x(d):
        d = date.fromisoformat(d)
        return L + (d - d0).days / (d1 - d0).days * (R - L)

    def y(v):
        return B - (v - y0) / (y1 - y0) * (B - T)

    lb = lad["leaderboard"]
    p = head(W, H, "Kaggle rating of each submission, July to August 2026",
             "Each dot is one submission's rating on the final ladder. Filled: first submission "
             "of a build; open: a resubmitted copy. Source: Kaggle submissions list, results.json.")
    p.append(text(24, 34, "Every build's rating on the live ladder", 20, weight="700"))
    p.append(text(24, 56, "TrueSkill rating of each submission on the final leaderboard. Filled dot: "
                          "a new build. Open dot: the same build resubmitted.", 14, SECONDARY))
    for v in range(500, 951, 100):
        dash = "" if v == 500 else ' stroke-dasharray="3 3"'
        p.append(f'<line x1="{L}" x2="{R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{RULE}" '
                 f'stroke-width="1"{dash}/>')
        p.append(text(L - 10, y(v) + 5, str(v), 12, SECONDARY, anchor="end"))
    for d, s in (("2026-07-01", "Jul 1"), ("2026-07-15", "Jul 15"), ("2026-08-01", "Aug 1"),
                 ("2026-08-10", "Aug 10")):
        p.append(text(x(d), B + 22, s, 12, SECONDARY, anchor="middle"))
    # team score line
    p.append(f'<line x1="{L}" x2="{R}" y1="{y(lb["score"]):.1f}" y2="{y(lb["score"]):.1f}" '
             f'stroke="{ORANGE}" stroke-width="2"/>')
    for i, line in enumerate(("Final leaderboard", f'{lb["score"]:.1f}',
                              f'rank {lb["rank"]:,} of {lb["teams"]:,}')):
        p.append(text(R + 12, y(lb["score"]) + 5 + 17 * i, line, 13, ORANGE, "700"))
    seen = set()
    for s in subs:
        if s["rating"] is None:
            continue
        cx, cy = x(s["submitted"]), y(s["rating"])
        first = s["build"] not in seen
        seen.add(s["build"])
        if first:
            p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{BLUE}"/>')
        else:
            p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{CARD}" stroke="{BLUE}" '
                     f'stroke-width="1.6"/>')
        if first and s["build"] in STACKED:
            rating, dx = STACKED[s["build"]]
            ly = y(rating)
            lx1 = cx + (6 if dx else 0)
            ly1 = cy if dx else cy - 7
            p.append(f'<line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{cx + dx:.1f}" y2="{ly + 5:.1f}" '
                     f'stroke="{SECONDARY}" stroke-width="0.8"/>')
            p.append(text(cx + dx - 4, ly, LABELS[s["build"]], 12.5, INK, "600", "start"))
        elif first and s["build"] in LABELS:
            above = s["build"] != "v11"
            p.append(text(cx + 9, cy + (-10 if above else 19), LABELS[s["build"]], 12.5, INK, "600"))
    n = lad["copies_per_build"].get("v12_pwr", 0)
    p.append(text(x("2026-08-01"), y(935), f"{n} copies of v12 (open dots)", 12.5, SECONDARY))
    p.append(text(24, H - 22, f"Ratings as listed on {lb['file'][-23:-13]}. Two submissions that "
                              "errored on Kaggle have no rating and are not shown.", 12, SECONDARY))
    p.append("</svg>")
    return "\n".join(p)


def noise_svg(res):
    cells = res["local_screens"]["cells"]
    rep = {}
    for c in cells:
        rep.setdefault((c["agent"], c["opponent"]), []).append(c)
    rows = []
    for (a, b), cs in rep.items():
        if len(cs) > 1:
            for i, c in enumerate(cs, 1):
                rows.append(("local", f"{a} vs {b}, run {i}", c["win_rate"], c["ci95"],
                             f'{c["wins"]}-{c["losses"]}'))
    m = res["matchups"]
    for sid, name in (("54842929", "v11"), ("55018765", "v12 (Powerful Hand fix)")):
        arch = m[sid]["by_archetype"]["Abra / Alakazam / Kadabra"]
        rows.append(("ladder", f"{name} vs Alakazam decks", arch["win_rate"], arch["ci95"],
                     f'{arch["wins"]}-{arch["games"] - arch["wins"]}'))
    W, label_w, plot_w = 920, 330, 440
    row_h, top = 30, 120
    groups = [("local", "Same matchup, played twice locally (204 games each)"),
              ("ladder", "Alakazam matchup on the live ladder, before and after the fix")]
    H = top + len(rows) * row_h + len(groups) * 44 + 40

    def x(v):
        return label_w + v / 100 * plot_w

    p = head(W, H, "Win rates with 95% intervals: local repeats and the live Alakazam matchup",
             "Dots are win rates; lines are 95% Wilson intervals. Source: tracked logs and cached "
             "ladder replays, results.json.")
    p.append(text(24, 34, "How much a win rate moves on its own", 20, weight="700"))
    p.append(text(24, 56, "Win rate with 95% Wilson interval. At 204 games the interval is about "
                          f"±{res['local_screens']['ci95_half_width_at_50pct_204_games']} points.",
                  14, SECONDARY))
    y = top - 30
    for key, title in groups:
        p.append(text(24, y + 14, title, 15, weight="700"))
        y += 30
        sub = [r for r in rows if r[0] == key]
        for tick in (0, 25, 50, 75, 100):
            dash = "" if tick == 50 else ' stroke-dasharray="3 3"'
            p.append(f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{y - 8}" '
                     f'y2="{y + len(sub) * row_h - 8}" stroke="{RULE}" stroke-width="1"{dash}/>')
        for _, name, wr, (lo, hi), wl in sub:
            cy = y + 6
            color = BLUE if key == "local" else PURPLE
            p.append(text(24, cy + 5, name, 13.5))
            p.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}" stroke="{color}" '
                     f'stroke-width="3" stroke-linecap="round"/>')
            p.append(f'<circle cx="{x(wr):.1f}" cy="{cy}" r="6" fill="{color}"/>')
            p.append(f'<text x="{label_w + plot_w + 16}" y="{cy + 5}" font-size="14" fill="{INK}">'
                     f'<tspan font-weight="700">{wr:.1f}%</tspan><tspan fill="{SECONDARY}" '
                     f'font-size="12"> {wl}, {lo:.0f}–{hi:.0f}</tspan></text>')
            y += row_h
        y += 14
    for tick in (0, 25, 50, 75, 100):
        p.append(text(x(tick), y + 4, f"{tick}%", 12, SECONDARY, anchor="middle"))
    p.append("</svg>")
    return "\n".join(p)


def main():
    res = json.load(open(os.path.join(REPO, "results", "results.json")))
    os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
    for name, svg in (("ladder", ladder_svg(res)), ("noise", noise_svg(res))):
        with open(os.path.join(REPO, "figures", f"{name}.svg"), "w") as f:
            f.write(svg + "\n")
        print(f"figures/{name}.svg")


if __name__ == "__main__":
    main()
