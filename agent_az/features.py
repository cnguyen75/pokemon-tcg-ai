"""Feature extraction shared by AZ training and inference.

STATE features come from value_features.extract_net (101 dims: 100 base + hc_tanh)
-- identical to what the value net and policy net already use, so we inherit the
proven encoding and the episodes.npz / policy.npz datasets line up.

OPTION features (38 dims) are the per-option encoding from train/extract_policy.
That logic lives in a training script; it is duplicated here (with the same
dict/dataclass-agnostic `_g` accessors) so agent_az has NO import dependency on
train/ at inference time. Kept byte-for-byte equivalent to extract_policy so the
mined policy.npz option rows and the live inference rows are the same distribution.
"""

from collections.abc import Sequence
import os
import sys
from typing import Any


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# agent2 ships value_features.py; add it to the path (inference copies it locally)
for _p in (os.path.join(_REPO, "agent2"), os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import AreaType, OptionType  # noqa: E402
import value_features as vf  # noqa: E402


N_OPT_TYPES = 17
D_OPT = 38
D_STATE = vf.N_NET_FEATURES  # 101


def _g(o: Any, k: str, d: Any = None) -> Any:
    v = o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
    return d if v is None else v


def _card_at(
    cur_players: Sequence[Any],
    me: int,
    area: AreaType | int,
    index: int,
    playerIndex: int | None,
    _CARD: dict[int, Any],
) -> Any:
    pi = playerIndex if playerIndex is not None else me
    try:
        pl = cur_players[pi]
        if area == AreaType.HAND:
            h = _g(pl, "hand")
            return h[index] if h else None
        if area == AreaType.ACTIVE:
            return (_g(pl, "active") or [None])[index]
        if area == AreaType.BENCH:
            return (_g(pl, "bench") or [])[index]
        if area == AreaType.DISCARD:
            return (_g(pl, "discard") or [])[index]
    except Exception:
        return None
    return None


def _card_static(cid: int, CARD: dict[int, Any]) -> list[float]:
    d = CARD.get(cid)
    if d is None:
        return [0.0] * 13
    ct = int(d.cardType)
    return [
        1.0 if ct == 0 else 0.0,
        1.0 if ct == 1 else 0.0,
        1.0 if ct == 3 else 0.0,
        1.0 if ct == 4 else 0.0,
        1.0 if ct in (5, 6) else 0.0,
        1.0 if ct == 2 else 0.0,
        1.0 if d.basic else 0.0,
        1.0 if d.stage1 else 0.0,
        1.0 if d.stage2 else 0.0,
        1.0 if d.ex else 0.0,
        1.0 if d.megaEx else 0.0,
        min(d.hp, 400) / 400.0,
        min(d.retreatCost, 4) / 4.0,
    ]


def option_features(
    o: Any, cur: Any, me: int, CARD: dict[int, Any], ATK: dict[int, Any]
) -> list[float]:
    """38-dim feature vector for one legal option. Works on replay dicts
    (training) or cg.api dataclasses (inference)."""
    t = int(_g(o, "type", -1))
    f = [1.0 if t == i else 0.0 for i in range(N_OPT_TYPES)]

    players = _g(cur, "players")
    my = players[me]
    op = players[1 - me]
    op_act = (_g(op, "active") or [None])[0] if op else None
    op_hp = _g(op_act, "hp", 0) if op_act is not None else 0
    op_id = _g(op_act, "id", 0) if op_act is not None else 0

    cid = 0
    area = _g(o, "area")
    index = _g(o, "index")
    pidx = _g(o, "playerIndex")
    if t in (
        OptionType.PLAY,
        OptionType.ATTACH,
        OptionType.EVOLVE,
        OptionType.ABILITY,
        OptionType.DISCARD,
        OptionType.CARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
    ):
        card = _card_at(players, me, area, index, pidx, CARD)
        cid = _g(card, "id", 0)
    f += _card_static(cid, CARD)

    atk_feats = [0.0, 0.0, 0.0, 0.0]
    if t == OptionType.ATTACK:
        aid = _g(o, "attackId", 0)
        my_act = (_g(my, "active") or [None])[0] if my else None
        att_id = _g(my_act, "id", 0) if my_act is not None else 0
        a = ATK.get(aid)
        if a is not None:
            dmg = vf._eff_damage(CARD.get(att_id), CARD.get(op_id), a)
            atk_feats[0] = min(dmg, 400) / 400.0
            atk_feats[1] = min(len(a.energies), 5) / 5.0
            atk_feats[2] = 1.0 if (op_hp and dmg >= op_hp) else 0.0
            n_prz = len(_g(op, "prize") or [])
            atk_feats[3] = (
                1.0 if (op_hp and dmg >= op_hp and n_prz <= vf._pyield(CARD.get(op_id))) else 0.0
            )
    f += atk_feats

    ipa = _g(o, "inPlayArea")
    f.append(1.0 if ipa == AreaType.ACTIVE else 0.0)
    f.append(1.0 if ipa == AreaType.BENCH else 0.0)
    f.append(1.0 if pidx == (1 - me) else 0.0)
    f.append(1.0)  # bias
    return f  # 38


def state_features(cur: Any, me: int, CARD: dict[int, Any], ATK: dict[int, Any]) -> list[float]:
    """101-dim state feature vector (value_features.extract_net)."""
    return vf.extract_net(cur, me, CARD, ATK)
