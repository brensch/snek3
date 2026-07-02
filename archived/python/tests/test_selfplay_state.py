"""Self-play state persistence for stopping and resuming in-flight games."""

import snek


def test_selfplay_state_save_load_public_api(tmp_path):
    state_path = tmp_path / "selfplay_state.bin"
    roundtrip_path = tmp_path / "selfplay_state_roundtrip.bin"

    state_id = snek.create_selfplay_state(board=11, num_snakes=4, count=6, seed=123)
    assert snek.save_selfplay_state(state_id, str(state_path)) is True
    assert state_path.stat().st_size > 0

    restored_id = snek.load_selfplay_state(str(state_path))
    assert isinstance(restored_id, int)
    info = snek.selfplay_state_info(restored_id)
    assert info["count"] == 6
    assert info["gpu_batch_games"] == 6
    assert info["shards"] == 1
    assert info["slots"] == 6
    assert info["nonempty_slots"] == 0
    assert info["pending_steps"] == 0
    assert info["pending_alive_samples"] == 0
    assert snek.save_selfplay_state(restored_id, str(roundtrip_path)) is True
    assert roundtrip_path.stat().st_size > 0
