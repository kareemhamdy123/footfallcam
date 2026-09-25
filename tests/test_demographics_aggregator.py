"""Aggregate demographics: counts, percentages and idempotency.

Percentages here are arithmetic over labels the classifier produced. Whether
those labels are any good is a separate question - see the warning at the top
of `features/gender.py`.
"""
import pytest

from features.gender import UNKNOWN, DemographicsAggregator
from tests.conftest import make_track


def track(tid, gender):
    t = make_track(tid)
    t.gender = gender
    return t


# ------------------------------------------------------------------ recording


def test_recording_a_classified_track_counts_it():
    agg = DemographicsAggregator()

    assert agg.record_track(track(1, "male")) == "male"
    assert agg.record_track(track(2, "female")) == "female"
    assert agg.counts == {"unknown": 0, "male": 1, "female": 1}


def test_a_track_is_only_counted_once_however_often_it_is_recorded():
    agg = DemographicsAggregator()
    person = track(1, "male")

    for _ in range(5):
        agg.record_track(person)

    assert agg.counts["male"] == 1, "one person is one person"


def test_unclassified_and_unusable_tracks_are_not_counted():
    agg = DemographicsAggregator()

    assert agg.record_track(track(1, None)) is None
    assert agg.record_track(track(2, UNKNOWN)) is None
    assert agg.record_track(track(3, "not-a-label")) is None
    assert agg.summary()["total_classified"] == 0


def test_recording_sales_conversations_is_a_separate_tally():
    agg = DemographicsAggregator()

    agg.record_track(track(1, "female"))
    agg.record_sales_conversation("female")
    agg.record_sales_conversation("female")
    agg.record_sales_conversation("male")

    summary = agg.summary()
    assert summary["counts"]["female"] == 1
    assert summary["sales_conversations"] == {"unknown": 0, "male": 1, "female": 2}


def test_sales_conversation_rejects_an_unknown_label():
    agg = DemographicsAggregator()
    assert agg.record_sales_conversation("nonsense") is None
    assert agg.sales_conversations["unknown"] == 0


# ---------------------------------------------------------------- percentages


def test_percentages_are_relative_to_the_classified_population():
    agg = DemographicsAggregator()
    for i in range(3):
        agg.record_track(track(i, "male"))
    for i in range(10, 13):
        agg.record_track(track(i, "female"))

    summary = agg.summary()
    assert summary["total_classified"] == 6
    assert summary["male_percentage"] == pytest.approx(50.0)
    assert summary["female_percentage"] == pytest.approx(50.0)


def test_an_even_split_is_exactly_one_hundred_percent():
    agg = DemographicsAggregator()
    agg.record_track(track(1, "male"))
    agg.record_track(track(2, "female"))

    summary = agg.summary()
    assert summary["male_percentage"] + summary["female_percentage"] == pytest.approx(100.0)


def test_an_all_male_population_reports_zero_percent_female():
    agg = DemographicsAggregator()
    agg.record_track(track(1, "male"))
    agg.record_track(track(2, "male"))

    summary = agg.summary()
    assert summary["male_percentage"] == pytest.approx(100.0)
    assert summary["female_percentage"] == pytest.approx(0.0)


def test_percentages_are_rounded_to_two_places():
    agg = DemographicsAggregator()
    for i in range(1):
        agg.record_track(track(i, "male"))
    for i in range(10, 13):  # 1 male, 3 female -> 25.0 / 75.0
        agg.record_track(track(i, "female"))

    summary = agg.summary()
    assert summary["male_percentage"] == 25.0
    assert summary["female_percentage"] == 75.0


def test_an_empty_aggregator_reports_zeroes_not_a_division_error():
    summary = DemographicsAggregator().summary()

    assert summary["total_classified"] == 0
    assert summary["male_percentage"] == 0.0
    assert summary["female_percentage"] == 0.0
    assert summary["counts"] == {"unknown": 0, "male": 0, "female": 0}


def test_unknowns_do_not_distort_the_percentage_split():
    """An 'unknown' is not a third gender; it is an absence of an answer."""
    agg = DemographicsAggregator()
    agg.record_track(track(1, "male"))
    agg.record_track(track(2, "female"))
    for i in range(10, 15):
        agg.record_track(track(i, None))  # never classified

    summary = agg.summary()
    assert summary["total_classified"] == 2
    assert summary["male_percentage"] == pytest.approx(50.0)
    assert summary["female_percentage"] == pytest.approx(50.0)


# ------------------------------------------------------------------- summary


def test_summary_exposes_the_whole_shape():
    summary = DemographicsAggregator().summary()
    for key in ("male_percentage", "female_percentage", "total_classified", "counts"):
        assert key in summary


def test_counts_are_a_copy_so_callers_cannot_corrupt_the_aggregate():
    agg = DemographicsAggregator()
    agg.record_track(track(1, "male"))

    summary = agg.summary()
    summary["counts"]["male"] = 999

    assert agg.counts["male"] == 1


def test_reset_clears_everything():
    agg = DemographicsAggregator()
    agg.record_track(track(1, "male"))
    agg.record_sales_conversation("male")

    agg.reset()

    assert agg.summary()["total_classified"] == 0
    assert agg.sales_conversations == {"unknown": 0, "male": 0, "female": 0}
    # Idempotency memory is cleared too, so the same id can be recorded again.
    assert agg.record_track(track(1, "male")) == "male"
