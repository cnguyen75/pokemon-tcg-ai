"""EXPERIMENT ONLY (does NOT touch agent2). Card-level feature extractor.

Adds a card-level representation on top of the existing 100 scalar features.
For a fixed vocabulary of card ids (the meta cards that actually appear), we
build per-zone count vectors so the net can see WHICH cards are where:

  zones (from player `me`'s perspective):
    my_hand        - our hand card ids (fully observed for us)
    my_board       - our active+bench: pokemon ids + attached energyCards ids
                     + tools ids + preEvolution ids
    my_discard     - our discard pile ids
    op_board       - opponent active+bench (+attachments/preEvo) - revealed
    op_discard     - opponent discard ids - revealed

Each zone -> a count vector of length (V + N_OTHER) where the last N_OTHER
slots are "other" buckets keyed by card category (pokemon/item/tool/supporter/
stadium/energy/unknown) for ids not in the vocab. Multi-hot/count, MLP-friendly.

build_vocab() computes the vocab from replay dirs; card_extract() produces
base(100) + card features. Everything is plain python + injected CARD/ATK db.
"""

import glob
import json
import os
import sys
from typing import Any


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))
import value_features as vf  # the 100-feat base extractor (== agent2 live)  # noqa: E402


# card categories for the "other" buckets (CardType enum in cg.api)
_CAT_NAMES = [
    "pokemon",
    "item",
    "tool",
    "supporter",
    "stadium",
    "basic_energy",
    "special_energy",
    "unknown",
]
N_OTHER = len(_CAT_NAMES)
_ZONES = ["my_hand", "my_board", "my_discard", "op_board", "op_discard"]


def _g(o: Any, k: str, d: Any = None) -> Any:
    if isinstance(o, dict):
        v = o.get(k, d)
    else:
        v = getattr(o, k, d)
    return d if v is None else v


def _cat_of(cid: int, CARD: dict[int, Any]) -> str:
    d = CARD.get(cid)
    if d is None:
        return "unknown"
    ct = int(_g(d, "cardType", -1))
    if 0 <= ct < len(_CAT_NAMES):
        return _CAT_NAMES[ct]
    return "unknown"


def _poke_ids_with_attach(ps: Any, CARD: dict[int, Any]) -> list[int]:
    """All revealed card ids on a player's board: pokemon + attached energy +
    tools + pre-evolution cards (active and bench)."""
    ids = []

    def add_poke(p: Any) -> None:
        if p is None:
            return
        ids.append(_g(p, "id", 0))
        for c in _g(p, "energyCards", []) or []:
            if c is not None:
                ids.append(_g(c, "id", 0))
        for c in _g(p, "tools", []) or []:
            if c is not None:
                ids.append(_g(c, "id", 0))
        for c in _g(p, "preEvolution", []) or []:
            if c is not None:
                ids.append(_g(c, "id", 0))

    for p in _g(ps, "active", []) or []:
        add_poke(p)
    for p in _g(ps, "bench", []) or []:
        add_poke(p)
    return ids


def _hand_ids(ps: Any) -> list[int]:
    return [_g(c, "id", 0) for c in (_g(ps, "hand", []) or []) if c is not None]


def _discard_ids(ps: Any) -> list[int]:
    return [_g(c, "id", 0) for c in (_g(ps, "discard", []) or []) if c is not None]


def zone_ids(cur: Any, me: int, CARD: dict[int, Any]) -> dict[str, list[int]]:
    """dict zone_name -> list of card ids for player `me`'s perspective."""
    players = _g(cur, "players")
    my, op = players[me], players[1 - me]
    return {
        "my_hand": _hand_ids(my),
        "my_board": _poke_ids_with_attach(my, CARD),
        "my_discard": _discard_ids(my),
        "op_board": _poke_ids_with_attach(op, CARD),
        "op_discard": _discard_ids(op),
    }


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def build_vocab(dirs: list[str], CARD: dict[int, Any], top_n: int = 300) -> list[int]:
    """Count card-id frequency across all replays in `dirs`, keep the top_n."""
    from collections import Counter

    cnt = Counter()
    files = []
    for dpath in dirs:
        files += sorted(glob.glob(os.path.join(dpath, "*.json")))
    for path in files:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict) or "steps" not in d:
            continue
        for step in d.get("steps", []):
            for j, ag in enumerate(step):
                if ag.get("status") != "ACTIVE":
                    continue
                obs = ag.get("observation") or {}
                cur = obs.get("current")
                if not cur:
                    continue
                for me in (0, 1):
                    for zids in zone_ids(cur, me, CARD).values():
                        cnt.update(zids)
    cnt.pop(0, None)
    top = [cid for cid, _ in cnt.most_common(top_n)]
    return sorted(top)


VOCAB_PATH = os.path.join(REPO, "train", "data", "card_vocab.json")


def load_vocab() -> list[int]:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def make_index(vocab: list[int]) -> dict[int, int]:
    return {cid: i for i, cid in enumerate(vocab)}


def card_feature_names(vocab: list[int], CARD: dict[int, Any]) -> list[str]:
    names = []
    for z in _ZONES:
        for cid in vocab:
            nm = _g(CARD.get(cid), "name", str(cid)) if CARD.get(cid) else str(cid)
            names.append(f"{z}#{cid}")
        for cat in _CAT_NAMES:
            names.append(f"{z}#other_{cat}")
    return names


def card_only_feats(
    cur: Any, me: int, CARD: dict[int, Any], vocab: list[int], idx: dict[int, int]
) -> list[float]:
    """Just the card-level part: len == len(_ZONES) * (len(vocab)+N_OTHER)."""
    V = len(vocab)
    W = V + N_OTHER
    out = [0.0] * (len(_ZONES) * W)
    zids = zone_ids(cur, me, CARD)
    for zi, z in enumerate(_ZONES):
        base = zi * W
        for cid in zids[z]:
            if cid == 0:
                continue
            pos = idx.get(cid)
            if pos is not None:
                out[base + pos] += 1.0
            else:
                cat = _cat_of(cid, CARD)
                out[base + V + _CAT_NAMES.index(cat)] += 1.0
    return out


def card_extract(
    cur: Any,
    me: int,
    CARD: dict[int, Any],
    ATK: dict[int, Any],
    vocab: list[int],
    idx: dict[int, int],
) -> list[float]:
    """base 100 features + squashed hc + card features (net input)."""
    f = vf.extract(cur, me, CARD, ATK)
    f.append(vf.hc_squash(vf.handcrafted_eval(cur, me)))
    f += card_only_feats(cur, me, CARD, vocab, idx)
    return f
