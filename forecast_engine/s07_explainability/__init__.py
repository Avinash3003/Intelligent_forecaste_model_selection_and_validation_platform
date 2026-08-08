"""SHAP / Feature Importance Explainability (Section 6.10).

Runs immediately after Forward Forecast Validation and before Model
Ranking: every surviving model's feature importance is generated once here
and stored, then consumed both by Ranking (as one of its composite inputs)
and later by the LLM Insight Engine (Section 6.12). Nothing in this package
produces narrative text — only structured importance values and stability
metadata.
"""
