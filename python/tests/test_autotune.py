from azsnek.autotune import TuneLimits, TuneSettings, tune_next


def test_tune_next_cuts_training_when_losses_regress():
    rows = [
        {"gen": 5, "policy_loss": 0.915, "value_loss": 0.190, "buffer": 300_000},
        {"gen": 6, "policy_loss": 0.920, "value_loss": 0.198, "buffer": 350_000},
        {"gen": 7, "policy_loss": 0.927, "value_loss": 0.210, "buffer": 400_000},
        {"gen": 8, "policy_loss": 0.935, "value_loss": 0.217, "buffer": 450_000},
        {
            "gen": 9,
            "policy_loss": 0.941,
            "value_loss": 0.223,
            "buffer": 450_000,
            "target_entropy": 0.41,
            "target_max_prob": 0.79,
        },
    ]
    settings = TuneSettings(samples=50_000, train_steps=1024, batch_size=2048)

    tuned, reasons = tune_next(settings, TuneLimits(), rows)

    assert tuned.train_steps < 300
    assert tuned.samples > settings.samples
    assert any("reduce optimization pressure" in r for r in reasons)


def test_tune_next_raises_tau_for_soft_targets():
    rows = [
        {
            "gen": 0,
            "policy_loss": 1.2,
            "value_loss": 0.3,
            "buffer": 50_000,
            "target_entropy": 0.94,
            "target_max_prob": 0.48,
        }
    ]
    settings = TuneSettings(samples=50_000, train_steps=256, tau=30.0)

    tuned, reasons = tune_next(settings, TuneLimits(), rows)

    assert tuned.tau > settings.tau
    assert any("raise tau" in r for r in reasons)
