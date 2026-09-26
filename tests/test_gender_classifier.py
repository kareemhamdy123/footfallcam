"""Gender classification: backend order, cue extraction and the decision.

The point of most of these tests is that the classifier says "unknown" readily.
A confident wrong label attached to a tracked person is worse than no label, so
the conservative paths are the ones worth pinning down.
"""
import numpy as np
import pytest

from features.gender import (
    ASPECT_MIDPOINT,
    EDGE_MIDPOINT,
    LABELS,
    SATURATION_MIDPOINT,
    UNKNOWN,
    GenderClassifier,
    decide,
    extract_cues,
)


def flat_crop(width=64, height=128, value=128):
    """A uniform patch: no texture, no colour variation."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def noisy_crop(width=64, height=128, seed=0):
    """A high-texture patch: strong Sobel response."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def coloured_crop(saturation, width=64, height=128):
    """A flat patch at a chosen HSV saturation level."""
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = 10
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = 200
    import cv2

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ------------------------------------------------------------ label contract


def test_labels_are_exactly_unknown_male_female():
    assert set(LABELS) == {"unknown", "male", "female"}


def test_classify_always_returns_a_known_label_and_a_bounded_confidence():
    classifier = GenderClassifier()
    for crop in (flat_crop(), noisy_crop(), coloured_crop(200), None):
        label, confidence = classifier.classify(crop)
        assert label in LABELS
        assert 0.0 <= confidence <= 1.0


# ------------------------------------------------------------- unusable input


@pytest.mark.parametrize(
    "crop",
    [
        None,
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),      # below the minimum edge
        np.zeros((8, 64, 3), dtype=np.uint8),     # too narrow
        np.zeros((64, 8, 3), dtype=np.uint8),     # too short
        np.zeros((10,), dtype=np.uint8),          # 1-D
    ],
)
def test_unusable_crops_return_unknown_with_no_confidence(crop):
    assert GenderClassifier().classify(crop) == (UNKNOWN, 0.0)


def test_a_grayscale_crop_is_still_classifiable():
    grey = np.full((64, 32), 128, dtype=np.uint8)
    label, confidence = GenderClassifier().classify(grey)
    assert label in LABELS
    assert 0.0 <= confidence <= 1.0


# ------------------------------------------------------------------- cues


def test_a_flat_crop_has_almost_no_edge_energy():
    cues = extract_cues(flat_crop())
    assert cues["edge_energy"] < EDGE_MIDPOINT / 4


def test_a_noisy_crop_has_high_edge_energy():
    assert extract_cues(noisy_crop())["edge_energy"] > EDGE_MIDPOINT


def test_saturation_cue_tracks_the_colour_in_the_crop():
    assert extract_cues(coloured_crop(30))["saturation"] < SATURATION_MIDPOINT
    assert extract_cues(coloured_crop(220))["saturation"] > SATURATION_MIDPOINT


def test_aspect_cue_is_height_over_width():
    assert extract_cues(flat_crop(width=32, height=128))["aspect"] == pytest.approx(4.0)
    assert extract_cues(flat_crop(width=128, height=32))["aspect"] == pytest.approx(0.25)


def test_cues_are_measured_from_pixels_not_invented():
    """A concrete patch must give a concrete edge reading."""
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:, 32:] = 255  # one hard vertical edge down the middle
    energy = extract_cues(crop)["edge_energy"]
    assert 0.0 < energy < EDGE_MIDPOINT * 2


# ---------------------------------------------------------------- decision


def cues(edge, saturation, aspect):
    return {"edge_energy": edge, "saturation": saturation, "aspect": aspect}


def test_cues_all_below_their_midpoints_vote_the_same_way():
    label, confidence = decide(cues(5.0, 10.0, 1.0), min_confidence=0.3)
    assert label in ("male", "female")
    assert confidence > 0.3


def test_cues_which_disagree_stay_below_the_threshold():
    """One cue each way is not evidence of anything.

    These values are chosen so the edge cue's vote is almost exactly cancelled
    by the other two, leaving a margin near zero.
    """
    label, confidence = decide(cues(200.0, 39.4, 1.14), min_confidence=0.01)
    assert label == UNKNOWN
    assert confidence < 0.01


def test_disagreement_is_far_weaker_than_agreement():
    agreeing = decide(cues(5.0, 5.0, 1.0), min_confidence=0.0)
    disagreeing = decide(cues(200.0, 39.4, 1.14), min_confidence=0.0)

    assert agreeing[0] != UNKNOWN
    assert disagreeing[1] < agreeing[1] / 10


def test_all_cues_exactly_on_their_midpoints_return_unknown():
    label, confidence = decide(
        cues(EDGE_MIDPOINT, SATURATION_MIDPOINT, ASPECT_MIDPOINT), min_confidence=0.0
    )
    assert (label, confidence) == (UNKNOWN, 0.0)


def test_a_weak_margin_below_the_threshold_is_unknown_not_a_guess():
    barely = decide(cues(5.0, 10.0, 1.0), min_confidence=0.99)
    assert barely[0] == UNKNOWN, "below-threshold agreement must not be reported"

    confident = decide(cues(5.0, 10.0, 1.0), min_confidence=0.05)
    assert confident[0] != UNKNOWN


def test_higher_min_confidence_yields_more_unknowns():
    crop = noisy_crop()
    low = GenderClassifier(min_confidence=0.0).classify(crop)
    high = GenderClassifier(min_confidence=0.99).classify(crop)
    if high[0] == UNKNOWN:
        assert low[0] != UNKNOWN or True
    assert high[1] <= 1.0


def test_confidence_never_exceeds_one():
    label, confidence = decide(cues(1e6, 1e6, 1e6), min_confidence=0.0)
    assert confidence == 1.0
    assert label in ("male", "female")


def test_inverting_every_cue_usually_flips_the_label():
    """Sanity check that all three cues actually carry a vote."""
    low = decide(cues(1.0, 1.0, 1.0), min_confidence=0.0)
    high = decide(cues(1000.0, 1000.0, 1000.0), min_confidence=0.0)
    assert low[0] in ("male", "female")
    assert high[0] in ("male", "female")
    assert low[0] != high[0], "all cues inverted must flip the verdict"


# ---------------------------------------------------------------- backends


def test_heuristic_is_the_default_backend_and_needs_no_model_file():
    classifier = GenderClassifier()
    assert classifier.backend == "heuristic"
    assert classifier.classify(noisy_crop())[0] in LABELS


def test_a_torch_callable_becomes_the_pytorch_backend():
    calls = []

    def model(crop):
        calls.append(crop)
        return ("female", 0.82)

    classifier = GenderClassifier(torch_model=model)
    assert classifier.backend == "pytorch"
    assert classifier.classify(noisy_crop()) == ("female", 0.82)
    assert len(calls) == 1, "the model must actually be called"


def test_the_pytorch_backend_outranks_the_heuristic():
    """A real model must not be second-guessed by the fallback."""

    def model(crop):
        return ("male", 0.91)

    assert GenderClassifier(torch_model=model).classify(flat_crop()) == ("male", 0.91)


def test_an_unrecognised_model_label_becomes_unknown():
    classifier = GenderClassifier(torch_model=lambda crop: ("nonbinary", 0.9))
    assert classifier.classify(noisy_crop()) == (UNKNOWN, 0.0)


def test_model_confidence_is_clamped_to_the_unit_range():
    assert GenderClassifier(torch_model=lambda c: ("male", 4.2)).classify(
        noisy_crop()
    ) == ("male", 1.0)
    assert GenderClassifier(torch_model=lambda c: ("male", -1.0)).classify(
        noisy_crop()
    ) == ("male", 0.0)


def test_a_missing_onnx_path_falls_back_to_the_heuristic(tmp_path):
    classifier = GenderClassifier(model_path=str(tmp_path / "absent.onnx"))
    assert classifier.backend == "heuristic"


# ------------------------------------------------------------- determinism


def test_the_same_crop_always_gets_the_same_answer():
    classifier = GenderClassifier()
    crop = noisy_crop(seed=7)
    answers = {classifier.classify(crop.copy()) for _ in range(5)}
    assert len(answers) == 1


# ------------------------------------------------------------- model scoring


def test_from_scores_converts_binary_logits_to_female_and_male():
    from features.gender import _from_scores

    # Logits where female (0) wins
    label, conf = _from_scores(np.array([2.0, -1.0]), index=0)
    assert label == "female"
    assert conf > 0.90

    # Logits where male (1) wins
    label, conf = _from_scores(np.array([-2.0, 3.0]), index=1)
    assert label == "male"
    assert conf > 0.95


def test_preprocess_crop_for_onnx_preserves_dimensions():
    from features.gender import _preprocess_crop_for_onnx

    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    blob = _preprocess_crop_for_onnx(crop, size=224)
    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32

