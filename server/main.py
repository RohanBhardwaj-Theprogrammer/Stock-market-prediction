from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uuid
import os
import json
from typing import Optional

from hybrid_model.model import HybridForecaster, ModelParams, load_dataset
from hybrid_model.tuning import random_search, SearchSpace
from .db import init_db, get_db, User, Preference, Run
from sqlalchemy.orm import Session
from pymongo import MongoClient

app = FastAPI(title="Hybrid LSTM+ARIMA Forecaster")

# Allow local dev client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static outputs
STATIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
os.makedirs(STATIC_ROOT, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

DATASET_MAP = {
    "dataset_01": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_01.csv")),
    "dataset_02": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_02.csv")),
}


@app.on_event("startup")
def _startup():
    init_db()
    # Prepare optional Mongo
    global _mongo_client, _mongo_collection
    _mongo_client = None
    _mongo_collection = None
    mongo_url = os.getenv("MONGO_URL")
    if mongo_url:
        try:
            _mongo_client = MongoClient(mongo_url)
            _mongo_collection = _mongo_client["forecasting"]["runs"]
        except Exception:
            _mongo_client = None
            _mongo_collection = None


def _log_run_mongo(doc: dict):
    try:
        if _mongo_collection is not None:
            _mongo_collection.insert_one(doc)
    except Exception:
        pass


class TrainRequest(BaseModel):
    dataset: str = Field(..., description="dataset_01 or dataset_02")
    days: int = 30
    n_lags: int = 30
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    number_nodes: int = 64
    acf_pacf_lags: int = 30


@app.post("/train")
async def train(req: TrainRequest, user_id: Optional[int] = None, db: Session = Depends(get_db)):
    if req.dataset not in DATASET_MAP:
        return {"error": f"Unknown dataset '{req.dataset}'. Use dataset_01 or dataset_02."}

    # prepare run folder
    run_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(STATIC_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    df = load_dataset(DATASET_MAP[req.dataset])
    params = ModelParams(
        n_lags=req.n_lags,
        days=req.days,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        number_nodes=req.number_nodes,
        acf_pacf_lags=req.acf_pacf_lags,
    )
    forecaster = HybridForecaster(df, output_dir=out_dir, params=params)
    result = forecaster.run()

    # return links to images and core metrics
    files = sorted([f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")])
    resp = {
        "run_id": run_id,
        "static_base": f"/static/{run_id}",
        "images": [f"/static/{run_id}/{name}" for name in files],
        "metrics": {
            "mse": result.mse,
            "rmse": result.rmse,
            "mae": result.mae,
            "arima_mse": result.arima_mse,
            "arima_rmse": result.arima_rmse,
            "arima_mae": result.arima_mae,
            "final_forecast_next": result.final_forecast_next,
        },
        "days": result.days,
    }

    # persist run (SQL + optional Mongo)
    try:
        run_row = Run(
            user_id=user_id,
            run_uid=run_id,
            dataset=req.dataset,
            params_json=json.dumps(req.model_dump()),
            metrics_json=json.dumps(resp["metrics"]),
        )
        db.add(run_row)
        db.commit()
    except Exception:
        db.rollback()

    _log_run_mongo({
        "kind": "train",
        "run_id": run_id,
        "user_id": user_id,
        "dataset": req.dataset,
        "params": req.model_dump(),
        "metrics": resp["metrics"],
    })

    return resp


@app.get("/")
async def root():
    return {"status": "ok"}


class TuneRequest(BaseModel):
    dataset: str
    trials: int = 10
    # Optional bounds overriding defaults
    n_lags_min: int | None = None
    n_lags_max: int | None = None
    epochs_min: int | None = None
    epochs_max: int | None = None
    lr_min: float | None = None
    lr_max: float | None = None


@app.post("/tune")
async def tune(req: TuneRequest):
    if req.dataset not in DATASET_MAP:
        return {"error": f"Unknown dataset '{req.dataset}'. Use dataset_01 or dataset_02."}
    df = load_dataset(DATASET_MAP[req.dataset])
    series = df["Adj_Close"].dropna()

    space = SearchSpace()
    if req.n_lags_min is not None and req.n_lags_max is not None:
        space.n_lags = (int(req.n_lags_min), int(req.n_lags_max))
    if req.epochs_min is not None and req.epochs_max is not None:
        space.epochs = (int(req.epochs_min), int(req.epochs_max))
    if req.lr_min is not None and req.lr_max is not None:
        space.learning_rate = (float(req.lr_min), float(req.lr_max))

    best = random_search(series, req.trials, space)
    doc = {"best_params": best["params"], "score_rmse": best["score"]}
    _log_run_mongo({
        "kind": "tune",
        "dataset": req.dataset,
        "trials": req.trials,
        "search_space": {
            "n_lags": getattr(space, "n_lags", None),
            "epochs": getattr(space, "epochs", None),
            "learning_rate": getattr(space, "learning_rate", None),
        },
        "result": doc,
    })
    return doc


# ---- Users & Preferences ----

class CreateUserRequest(BaseModel):
    username: str


@app.post("/users")
async def create_user(req: CreateUserRequest, db: Session = Depends(get_db)):
    # create if not exists
    user = db.query(User).filter(User.username == req.username).first()
    if user:
        return {"user_id": user.id, "username": user.username}
    user = User(username=req.username)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user_id": user.id, "username": user.username}


class SavePreferenceRequest(BaseModel):
    user_id: int
    params: dict


@app.post("/preferences")
async def save_preferences(req: SavePreferenceRequest, db: Session = Depends(get_db)):
    pref = db.query(Preference).filter(Preference.user_id == req.user_id).first()
    if pref is None:
        pref = Preference(user_id=req.user_id, params_json=json.dumps(req.params))
        db.add(pref)
    else:
        pref.params_json = json.dumps(req.params)
    db.commit()
    return {"status": "saved"}


@app.get("/preferences/{user_id}")
async def get_preferences(user_id: int, db: Session = Depends(get_db)):
    pref = db.query(Preference).filter(Preference.user_id == user_id).first()
    if not pref:
        return {"params": None}
    return {"params": json.loads(pref.params_json)}


@app.get("/runs")
async def list_runs(user_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Run)
    if user_id is not None:
        q = q.filter(Run.user_id == user_id)
    rows = q.order_by(Run.created_at.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "run_uid": r.run_uid,
            "dataset": r.dataset,
            "params": json.loads(r.params_json),
            "metrics": json.loads(r.metrics_json),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
