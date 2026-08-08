"""MLflow Experiment Tracking & Model Registry (Section 6.13).

Wraps every forecasting pipeline execution in one MLflow Parent Run,
logging the parameters, metrics, and artifacts already captured in the
run's `PipelineResult`, and registering the Final Production Model for
every forecasting group that has one.

`forecast_engine/tracking/mlflow_client.py` is the *only* module in this
package (and the only one in the whole engine) that imports the `mlflow`
SDK directly — every other module here works against that thin facade, so
retargeting Local Development to Databricks is entirely a
`MLflowConfig.tracking_uri` change, never a code change.
"""
