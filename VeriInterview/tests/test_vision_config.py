"""Sanity checks for the M3 config blocks (task G-4)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import config


def test_deepfake_weights_sum_to_one():
    w = config.DEEPFAKE_WEIGHTS
    assert abs(w.identity + w.sync + w.eye_authenticity - 1.0) < 1e-9


def test_single_signal_cap_stays_below_high():
    # one input alone must never reach HIGH (golden rule), except the explicit identity-switch floor
    assert config.DEEPFAKE_WEIGHTS.single_signal_cap <= config.RISK_THRESHOLDS.medium_max
    assert config.DEEPFAKE_WEIGHTS.strong_identity_switch_floor > config.RISK_THRESHOLDS.medium_max


def test_threshold_orderings_are_consistent():
    s, i, g = config.SYNC_THRESHOLDS, config.IDENTITY_THRESHOLDS, config.GAZE_THRESHOLDS
    assert s.bonafide_offset_frames < s.high_offset_frames <= s.vshift
    assert s.lse_c_low < s.lse_c_good
    assert i.below_threshold_medium_pct < i.below_threshold_high_pct
    assert g.eye_x_ratio_left < 1.0 < g.eye_x_ratio_right
    assert g.head_std_elevated_deg < g.head_std_high_deg


def test_vad_default_matches_media_module():
    import inspect
    from backend.vision import media
    default = inspect.signature(media.detect_speech).parameters["rel_threshold"].default
    assert default == config.VAD_SETTINGS.rel_threshold


def test_model_and_feature_dirs_are_gitignored_locations():
    assert config.MODELS_DIR.endswith(os.path.join("storage", "models"))
    assert config.FEATURES_DIR.endswith(os.path.join("storage", "features"))
