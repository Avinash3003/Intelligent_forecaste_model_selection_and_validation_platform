"""LLM Business Insights (Section 6.12).

Converts a finished pipeline run's outputs into business-readable
explanations. Strictly descriptive: this package never makes a forecasting
decision, never selects a model, and never feeds back into any earlier
stage — it consumes one standardized `PipelineResult` and produces
narrative text only.
"""
