import os
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


# Ensure mlruns directory exists and set a proper file:/// URI (works on Windows)
mlruns_path = Path(os.path.abspath("mlruns"))
mlruns_path.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(mlruns_path.as_uri())

def start_run():
    mlflow.set_experiment("YOLO_Streamlit")
    run = mlflow.start_run()
    return run

def log_hyperparams(params: dict, run_id: str | None = None):
    run = mlflow.active_run()
    if run is not None:
        for k, v in params.items():
            mlflow.log_param(k, v)
        return
    if run_id is None:
        raise RuntimeError("No active MLflow run and no run_id provided.")
    client = MlflowClient()
    for k, v in params.items():
        client.log_param(run_id, k, str(v))



def _resolve_run_id_or_latest(run_id: str | None = None) -> str:
    run = mlflow.active_run()
    if run:
        return run.info.run_id
    if run_id:
        return run_id
    client = MlflowClient()
    exp = client.get_experiment_by_name("YOLO_Streamlit")
    if exp is None:
        raise RuntimeError("No active run and no experiment 'YOLO_Streamlit' found.")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found in experiment 'YOLO_Streamlit'. Start a run first.")
    return runs[0].info.run_id

def log_metrics(metrics: dict, run_id: str | None = None):
    rid = _resolve_run_id_or_latest(run_id)
    run = mlflow.active_run()
    if run is not None and run.info.run_id == rid:
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        return
    client = MlflowClient()
    for k, v in metrics.items():
        client.log_metric(rid, k, float(v))


def log_artifact(file_path: str, artifact_path: str = None, run_id: str | None = None):
    abs_path = str(Path(file_path).resolve())
    if not Path(abs_path).is_file():
        raise FileNotFoundError(f"Artifact not found: {abs_path}")

    run = mlflow.active_run()
    if run is not None:
        # normal path
        if artifact_path:
            mlflow.log_artifact(abs_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(abs_path)
        return

    # fallback to client-based logging using run_id
    if run_id is None:
        raise RuntimeError("No active MLflow run. Call start_run() or pass run_id to log_artifact().")
    client = MlflowClient()
    client.log_artifact(run_id, abs_path, artifact_path=artifact_path)


def log_detection_history(detection_history):
    counts = [len(d['detections']) for d in detection_history]
    plt.figure()
    plt.plot(counts)
    plt.title("Detection history")
    plt.xlabel("Frame index")
    plt.ylabel("Objects detected")
    out = "detection_history.png"
    plt.savefig(out)
    log_artifact(out, artifact_path="plots")
    plt.close()
