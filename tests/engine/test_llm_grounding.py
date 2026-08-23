"""Deterministic grounding / faithfulness check (Section 13.1)."""

from forecast_engine.s11_llm.grounding import check_grounding, extract_numeric_facts


def test_a_correctly_cited_number_is_grounded():
    metrics = {"wmape": 25.86, "confidence_estimate": 74.1}
    result = check_grounding("xgboost had a WMAPE of 25.86%, off by about 26% on average.", metrics)
    assert result.grounded
    assert result.issues == []


def test_a_fabricated_number_is_flagged():
    metrics = {"wmape": 25.86, "confidence_estimate": 74.1}
    result = check_grounding("xgboost had a WMAPE of 91.2%, an excellent result.", metrics)
    assert not result.grounded
    assert any("91.2" in issue for issue in result.issues)


def test_rounding_is_tolerated_not_flagged():
    metrics = {"wmape": 25.859}
    result = check_grounding("WMAPE was about 26%.", metrics)
    assert result.grounded


def test_text_with_no_numbers_is_trivially_grounded():
    result = check_grounding("The model was selected on the evidence available.", {"wmape": 25.86})
    assert result.grounded
    assert result.checked_numbers == 0


def test_small_integers_are_not_treated_as_metric_claims():
    # Counts (1 rejected candidate, rank 2, horizon T12) should not need to
    # match a metrics fact — they aren't the kind of claim this exists to check.
    result = check_grounding("1 candidate was rejected before rank 2 was tried.", {"wmape": 25.86})
    assert result.grounded


def test_empty_text_is_grounded():
    result = check_grounding("", {"wmape": 25.86})
    assert result.grounded
    assert result.checked_numbers == 0


def test_no_facts_at_all_does_not_block_output():
    # A fresh fallback group may have no backtest metrics whatsoever; a
    # cited number cannot be confirmed OR refuted, so it must not be
    # treated as a hallucination by default.
    result = check_grounding("Confidence was 94.0%.", {"wmape": None, "confidence_estimate": None})
    assert result.grounded


def test_fraction_and_percent_scale_are_both_matched():
    # drift_statistic style values are often fraction-scale (0.43), while a
    # narrative may write it as a percent-like figure or vice versa.
    metrics = {"drift_statistic": 0.4318}
    result = check_grounding("The drift statistic was 43.18.", metrics)
    assert result.grounded


def test_extract_numeric_facts_includes_percent_and_fraction_forms():
    facts = extract_numeric_facts({"wmape": 8.2})
    assert 8.2 in facts
    assert any(abs(f - 0.082) < 1e-6 for f in facts)


def test_multiple_ungrounded_numbers_are_all_reported():
    metrics = {"wmape": 10.0}
    result = check_grounding("WMAPE was 55% and confidence was 999%.", metrics)
    assert not result.grounded
    assert len(result.issues) == 2
