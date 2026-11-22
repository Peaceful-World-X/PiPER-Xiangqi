"""Reference-film ownership, indexed by final board cell as in motion.npz."""


def reference_motion(motion):
    """Return the reference take with a private plan and original exchange marker IDs.

    Times are seconds in the recording. For the 11.6 s and 51 s exchanges
    the receiving arm carries straight to the final cell. The film briefly
    parks both tokens at the river for a third pickup, which the six-column
    plan cannot encode.
    """
    result = dict(motion)
    plan = result["plan"].copy()
    result["plan"] = plan
    left, right = 0, 1

    # Columns: donor, receiver, scattered source, pickup, handover, placement.
    # 11.6 s: the left hand feeds the advisor. The rook from source 5 is a
    # separate left-hand placement, visible on the corner by 13.95 s.
    plan[3] = left, right, 2, 9.45, 11.6, 15.8
    plan[0] = left, left, 5, 13.05, -1, 13.8

    # 51 s: the right hand feeds the pawn. The right-side cannon is picked from
    # source 21 after this exchange. The left-side cannon is already down at 50 s.
    plan[28] = right, left, 19, 48.6, 51.0, 55.4
    plan[21] = right, right, 21, 52.25, -1, 53.4
    plan[18] = left, left, 10, 49.15, -1, 49.8

    # 69.3 s: the right hand lifts the near-base disc, and the left places
    # it directly on the pawn cell by 71.2 s.
    plan[23] = right, left, 27, 67.45, 69.3, 71.1

    # Reconcile the two scatter piles. Some pickups in the film are occluded
    # or come from the river. These substitutions preserve unique sources and
    # top-first stack access. Only the source changes on the following rows.
    plan[10, 2] = 3
    plan[12, 2] = 7
    plan[7, 2] = 4
    plan[6, 2] = 15  # lower token, after source 10 on piece 18
    plan[26, 2] = 16
    plan[31, 2] = 24
    plan[17, 2] = 18  # bottom of sources 17 / 31 / 18

    # fit_exchange and the editorial pass anchors refer to the filmed beat.
    result["exchange_marker"] = {
        10: 10, 3: 0, 26: 26, 20: 20, 12: 12, 28: 21, 4: 4, 23: 23,
    }
    return result
