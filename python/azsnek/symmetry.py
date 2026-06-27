"""D4 (dihedral) symmetry augmentation for egocentric observations.

The board has an 8-element symmetry group (4 rotations x 2 reflections). Because
the egocentric canvas is odd-sized and head-centred, every D4 transform is exact
about the centre, so we can rotate/reflect the observation planes and get another
perfectly valid position. Augmenting training with these multiplies effective
data ~8x and bakes the symmetry prior into the net (as AlphaGo Zero did).

The subtlety is the policy target: the 4 move logits must be permuted to match
the board transform. We derive the array op and the move permutation from the
SAME (a, b, t) generators, so they are consistent by construction (verified by
`_self_test`).

Obs layout is [..., C, H=y, W=x]. Move order (snek-core): Up=+y, Down=-y,
Left=-x, Right=+x  ->  offsets in (drow=dy, dcol=dx):
    Up=(+1,0)  Down=(-1,0)  Left=(0,-1)  Right=(0,+1)
"""

from __future__ import annotations

import numpy as np

# Move offsets in (drow, dcol) = (dy, dx), indexed Up, Down, Left, Right.
_MOVE_OFFSETS = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=int)

# D4 generators as 2x2 matrices on (drow, dcol) offsets.
_A = np.array([[-1, 0], [0, 1]])   # flip rows  (np.flip axis=-2)
_B = np.array([[1, 0], [0, -1]])   # flip cols  (np.flip axis=-1)
_T = np.array([[0, 1], [1, 0]])    # transpose  (swapaxes -2,-1)


def _move_perm(M: np.ndarray) -> np.ndarray:
    """Permutation `perm` such that move m maps to move perm[m] under M."""
    perm = np.empty(4, dtype=np.int64)
    for m, off in enumerate(_MOVE_OFFSETS):
        mapped = M @ off
        match = np.where((_MOVE_OFFSETS == mapped).all(axis=1))[0]
        assert match.size == 1, f"offset {mapped} is not a unit move"
        perm[m] = match[0]
    return perm


def _build_transforms():
    """The 8 D4 elements as (apply_obs, move_perm). apply_obs acts on a
    [..., C, H, W] array; move_perm is a length-4 permutation of move indices."""
    transforms = []
    for a in (0, 1):
        for b in (0, 1):
            for t in (0, 1):
                # Array op order: flip rows, then flip cols, then transpose.
                # Coordinate map on offsets is therefore M = T^t @ B^b @ A^a.
                M = np.eye(2, dtype=int)
                if a:
                    M = _A @ M
                if b:
                    M = _B @ M
                if t:
                    M = _T @ M
                perm = _move_perm(M)

                def apply_obs(x, a=a, b=b, t=t):
                    if a:
                        x = np.flip(x, axis=-2)
                    if b:
                        x = np.flip(x, axis=-1)
                    if t:
                        x = np.swapaxes(x, -2, -1)
                    return np.ascontiguousarray(x)

                transforms.append((apply_obs, perm))
    return transforms


TRANSFORMS = _build_transforms()  # 8 (apply_obs, move_perm) pairs


def augment_batch(obs: np.ndarray, pol: np.ndarray, rng: np.random.Generator):
    """Apply one random D4 transform (shared across the batch) to obs and pol.

    obs: [B, C, H, W]; pol: [B, 4]. Value targets are symmetry-invariant and
    untouched. Returns (obs_t, pol_t) as contiguous float32 arrays.
    """
    apply_obs, perm = TRANSFORMS[rng.integers(len(TRANSFORMS))]
    obs_t = apply_obs(obs).astype(np.float32, copy=False)
    pol_t = np.empty_like(pol)
    pol_t[:, perm] = pol  # move m's mass moves to position perm[m]
    return obs_t, pol_t


def _self_test():
    """Verify each transform's array op and move permutation agree: a marker one
    cell in move-direction m from the centre must land one cell in direction
    perm[m] after the transform."""
    side = 7
    c = side // 2
    for apply_obs, perm in TRANSFORMS:
        for m, off in enumerate(_MOVE_OFFSETS):
            grid = np.zeros((1, 1, side, side), dtype=np.float32)
            grid[0, 0, c + off[0], c + off[1]] = 1.0
            out = apply_obs(grid)
            exp = _MOVE_OFFSETS[perm[m]]
            assert out[0, 0, c + exp[0], c + exp[1]] == 1.0, (perm, m)
        assert sorted(perm.tolist()) == [0, 1, 2, 3], perm
    return True


if __name__ == "__main__":
    _self_test()
    print("symmetry self-test OK")
