import datetime
import io
import json
import logging
import time
import zipfile
from typing import Any, List, Optional

import pandas as pd
import requests

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score, MAPE
from evidently.presets import DataDriftPreset

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge
from sklearn.ensemble import RandomForestRegressor

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Global Variables for Model and Data ---
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
DTEDAY_COL = 'dteday'

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
REFERENCE_START = "2011-01-01 00:00:00"
REFERENCE_END = "2011-01-31 23:00:00"

# --- Prometheus Metrics Definitions ---
registry = CollectorRegistry()

api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

model_rmse_score = Gauge(
    'model_rmse_score',
    'Root Mean Squared Error of the regression model on the latest evaluation batch',
    registry=registry
)

model_mae_score = Gauge(
    'model_mae_score',
    'Mean Absolute Error of the regression model on the latest evaluation batch',
    registry=registry
)

model_r2_score = Gauge(
    'model_r2_score',
    'R2 Score of the regression model on the latest evaluation batch',
    registry=registry
)

# --- Métrique de choix : score de dérive des données (Evidently DataDriftPreset) ---
# Justification : au-delà des scores de qualité du modèle (RMSE/MAE/R2), il est essentiel
# de savoir SI les features en entrée (meteo, saison...) ont dérivé par rapport à la
# référence (janvier 2011). Un score de dérive élevé peut expliquer une dégradation du
# RMSE avant même qu'elle ne devienne critique, et guide la décision de ré-entraînement.
model_data_drift_score = Gauge(
    'model_data_drift_score',
    'Share of input features detected as drifted vs reference (Evidently DataDriftPreset), between 0 and 1',
    registry=registry
)

# Bonus : MAPE, exposé car demandé dans EvaluationReportOutput
model_mape_score = Gauge(
    'model_mape_score',
    'Mean Absolute Percentage Error of the regression model on the latest evaluation batch',
    registry=registry
)

# --- Data Ingestion and Preparation Functions ---
def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    logger.info("Fetching data from UCI archive...")
    content = requests.get(DATASET_URL, verify=False, timeout=60).content
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        df = pd.read_csv(z.open("hour.csv"), header=0, sep=',', parse_dates=[DTEDAY_COL])
    logger.info("Data fetched successfully (%d rows).", len(df))
    return df


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processes raw data, setting a DatetimeIndex."""
    logger.info("Processing raw data...")
    raw_data['hr'] = raw_data['hr'].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(row[DTEDAY_COL].date(), datetime.time(row.hr)),
        axis=1
    )
    raw_data = raw_data.sort_index()
    logger.info("Data processed successfully.")
    return raw_data


def _train_and_predict_reference_model(processed_data: pd.DataFrame):
    """Trains the RandomForestRegressor on January 2011 data and returns
    the trained model + the reference DataFrame (with predictions added)."""
    logger.info("Training reference model on January 2011 data...")
    reference_data = processed_data.loc[REFERENCE_START:REFERENCE_END].copy()

    X_train = reference_data[NUM_FEATS + CAT_FEATS]
    y_train = reference_data[TARGET]

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_model.fit(X_train, y_train)

    reference_data[PREDICTION] = reg_model.predict(X_train)
    logger.info("Reference model trained on %d samples.", len(reference_data))
    return reg_model, reference_data


def _recursive_find_metric(obj, keyword: str):
    """Recursively searches an Evidently result dict for a metric whose
    identifier contains `keyword` (case-insensitive) and returns its value.
    Some Evidently metrics (MAE, MAPE) return {"mean": ..., "std": ...} instead
    of a plain float, and DriftedColumnsCount returns {"count": ..., "share": ...}.
    In those cases we extract the most relevant scalar automatically."""
    if isinstance(obj, dict):
        identifier = obj.get("metric_id") or obj.get("id") or obj.get("name")
        if identifier and keyword.lower() in str(identifier).lower() and "value" in obj:
            val = obj["value"]
            if isinstance(val, dict):
                for key in ("mean", "share", "count"):
                    if key in val:
                        return val[key]
                return None
            return val
        for v in obj.values():
            result = _recursive_find_metric(v, keyword)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _recursive_find_metric(item, keyword)
            if result is not None:
                return result
    return None


def _sanitize_float(value):
    """Returns None if value is NaN/Inf or not a valid finite number,
    otherwise returns the float value. Prevents JSON serialization errors
    and invalid Prometheus Gauge values."""
    if value is None:
        return None
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return None
    if f_value != f_value or f_value in (float("inf"), float("-inf")):  # NaN check
        return None
    return f_value


# --- Model Loading at Startup ---
try:
    _raw_data = _fetch_data()
    _processed_data = _process_data(_raw_data)
    model, reference_data = _train_and_predict_reference_model(_processed_data)
    logger.info("Model and reference dataset ready.")
except Exception as e:
    logger.error(f"Error during startup data/model preparation: {e}")
    raise RuntimeError("Failed to prepare model at startup, application cannot start.") from e


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")


class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)


class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput):
    start_time = time.time()
    status_code = "200"
    try:
        features = {feat: getattr(input_data, feat) for feat in NUM_FEATS + CAT_FEATS}
        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        return PredictionOutput(predicted_count=float(prediction))
    except Exception as e:
        status_code = "500"
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        duration = time.time() - start_time
        api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()


@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(payload: EvaluationData):
    start_time = time.time()
    status_code = "200"
    try:
        if not payload.data:
            status_code = "400"
            raise HTTPException(status_code=400, detail="No data provided for evaluation.")

        current_df = pd.DataFrame(payload.data)

        for col in NUM_FEATS:
            current_df[col] = pd.to_numeric(current_df[col], errors="coerce")
        for col in CAT_FEATS:
            current_df[col] = pd.to_numeric(current_df[col], errors="coerce")
        current_df[TARGET] = pd.to_numeric(current_df[TARGET], errors="coerce")
        current_df = current_df.dropna(subset=NUM_FEATS + CAT_FEATS + [TARGET])

        if current_df.empty:
            status_code = "400"
            raise HTTPException(status_code=400, detail="No valid rows after cleaning input data.")

        current_df[PREDICTION] = model.predict(current_df[NUM_FEATS + CAT_FEATS])

        data_definition = DataDefinition(
            numerical_columns=NUM_FEATS,
            categorical_columns=CAT_FEATS,
            regression=[Regression(target=TARGET, prediction=PREDICTION)],
        )
        current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)
        reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)

        report = Report([
            RMSE(), MAE(), R2Score(), MAPE(),
            DataDriftPreset(columns=NUM_FEATS + CAT_FEATS),
        ])
        my_eval = report.run(current_dataset, reference_dataset)
        report_dict = my_eval.dict()

        # DEBUG (à décommenter une fois pour inspecter la structure réelle si besoin) :
        # logger.info(json.dumps(report_dict, default=str)[:6000])

        rmse_value = _sanitize_float(_recursive_find_metric(report_dict, "RMSE"))
        mae_value = _sanitize_float(_recursive_find_metric(report_dict, "MAE"))
        r2_value = _sanitize_float(_recursive_find_metric(report_dict, "R2Score"))
        mape_value = _sanitize_float(_recursive_find_metric(report_dict, "MAPE"))

        drift_share = (
            _recursive_find_metric(report_dict, "DriftedColumnsCount")
            or _recursive_find_metric(report_dict, "DatasetDrift")
            or _recursive_find_metric(report_dict, "ShareOfDrifted")
        )
        drift_share = drift_share if drift_share is not None else 0.0
        drift_detected = 1 if drift_share > 0.5 else 0

        if rmse_value is not None:
            model_rmse_score.set(rmse_value)
        if mae_value is not None:
            model_mae_score.set(mae_value)
        if r2_value is not None:
            model_r2_score.set(r2_value)
        if mape_value is not None:
            model_mape_score.set(mape_value)
        model_data_drift_score.set(drift_share)

        logger.info(
            f"Evaluation '{payload.evaluation_period_name}' on {len(current_df)} items. "
            f"RMSE={rmse_value}, MAE={mae_value}, R2={r2_value}, MAPE={mape_value}, drift_share={drift_share}"
        )

        return EvaluationReportOutput(
            message=f"Evaluation completed for period '{payload.evaluation_period_name}'.",
            rmse=rmse_value,
            mape=mape_value,
            mae=mae_value,
            r2score=r2_value,
            drift_detected=drift_detected,
            evaluated_items=len(current_df),
        )
    except HTTPException:
        raise
    except Exception as e:
        status_code = "500"
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    finally:
        duration = time.time() - start_time
        api_request_duration_seconds.labels(endpoint="/evaluate", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/evaluate", method="POST", status_code=status_code).inc()


@app.get("/metrics")
async def metrics(request: Request):
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(registry), media_type="text/plain")
