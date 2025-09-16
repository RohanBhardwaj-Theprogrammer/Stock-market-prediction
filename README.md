## Hyperparameter Tuning

You can run a simple random search over LSTM hyperparameters before training.

- API endpoint: `POST /tune`
- Body:

```json
{
  "dataset": "dataset_01",
  "trials": 10,
  "n_lags_min": 15,
  "n_lags_max": 60,
  "epochs_min": 5,
  "epochs_max": 20,
  "lr_min": 0.0001,
  "lr_max": 0.005
}
```

- Response:

```json
{ "best_params": { "n_lags": 34, "epochs": 9, "learning_rate": 0.0013, "batch_size": 32, "number_nodes": 64 }, "score_rmse": 12.34 }
```

### Client flow
- In `client/index.html`, use "Tune then Train" with a number of trials; the client will run tuning, fill the form with chosen params, and launch training automatically.
- Tuning uses 20% of the series as validation horizon by default and minimizes RMSE.

## Users, Preferences, and History
- Backend persists users, saved preferences, and run history in a local SQLite database (`server/app.db`).
- Endpoints:
  - `POST /users` { "username": "alice" } → returns `{ user_id, username }`
  - `POST /preferences` { "user_id", "params": { ...train params... } } → saves
  - `GET /preferences/{user_id}` → returns saved params or null
  - `GET /runs?user_id=123` → recent runs for the user (or all if not specified)
- Optional MongoDB logging: set `MONGO_URL` to log run and tuning documents to `forecasting.runs`.

### Example (PowerShell)
```powershell
$env:MONGO_URL = "mongodb://localhost:27017"
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```
