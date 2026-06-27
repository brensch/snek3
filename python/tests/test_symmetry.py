import numpy as np

from azsnek.symmetry import TRANSFORMS, _self_test, augment_batch


def test_transforms_are_consistent():
    # Array op and move permutation agree, and there are all 8 D4 elements.
    assert _self_test()
    assert len(TRANSFORMS) == 8


def test_augment_preserves_shapes_and_policy_mass():
    rng = np.random.default_rng(0)
    obs = rng.random((5, 9, 21, 21)).astype(np.float32)
    pol = rng.random((5, 4)).astype(np.float32)
    pol /= pol.sum(axis=1, keepdims=True)
    obs_t, pol_t = augment_batch(obs, pol, rng)
    assert obs_t.shape == obs.shape
    assert pol_t.shape == pol.shape
    # Policy is a permutation of the original per row -> mass and sorted values preserved.
    np.testing.assert_allclose(pol_t.sum(axis=1), pol.sum(axis=1), rtol=1e-6)
    np.testing.assert_allclose(np.sort(pol_t, axis=1), np.sort(pol, axis=1), rtol=1e-6)
