# Energy Consumption Forecasting for Smart Grids

[![CI Pipeline](https://github.com/JoannaKusy/ML_for_Data_Analytics_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/JoannaKusy/ML_for_Data_Analytics_Project/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-green.svg)](https://github.com/astral-sh/ruff)

*Authors: Joanna Kusy, Karolina Pyś, Tomasz Srebniak, Romain Steffen  \
Wroclaw University of Science and Technology*
## Project Overview & Motivation
Accurate multivariate forecasting is critical for modern Smart Grids. Intelligent electricity networks require a delicate supply/demand balance, especially with the integration of renewable energy sources and the electrification of heating (Heat Pumps) and transport (EVs). 

This project implements an end-to-end Machine Learning pipeline to predict residential electricity demand (Grid Import) using both classical statistical methods and advanced Deep Learning architectures. It culminates in a deeply interpretable model evaluation and a fully deployed, real-time MLOps monitoring dashboard.

---

## Data Pipeline & Feature Engineering
The data pipeline is fully reproducible, taking raw multi-source data and transforming it into model-ready sequences. The research logic is documented in the notebooks, while the production pipeline is modularized in the `src/` directory.

**1. Raw Data Acquisition**
* **Open Power System Data (OPSD)** the `Household Data`, `Time series` and `Weather Data` datasets from the [following link](https://open-power-system-data.org/?fbclid=IwY2xjawQrXChleHRuA2FlbQIxMABicmlkETAySk9Rek9iRlhadG5obFY2c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHrxVocPQPnX3w-NtGDF0CzSaMXon2ozmHkI4GrpsZ26X0-5tdyYfNkCLztn8_aem_VLIgUHUoLSYcNf8ToLLKHw).


**2. Exploratory Data Analysis & Cleaning (`notebooks/01_eda.ipynb` )**
* Validates time-series continuity, handles missing value imputation and standardizes UTC timestamps. Outputs are saved to `data/cleaned/`.

**3. Advanced Feature Engineering (`notebooks/02_feature_engineering.ipynb`)**
* **Target Variable:** Energy Demand (Grid Import) in kWh.
* **Appliance-Level Regressors:** Heat Pump, Washing Machine, Dishwasher, EV, Photovoltaic (PV).
* **Weather & Solar Regressors:** Temperature, Radiation (Direct & Diffuse Horizontal). We utilize the `astral` library to calculate exact dawn/dusk times for solar impact analysis.
* **Temporal Regressors:** Incorporates the `holidays` library for regional holiday flags, alongside weekends, seasons, and cyclical time-of-day encodings.
* The final scaled and sequence-encoded data is exported to `data/processed/` for model training.

**Complete data pipeline based on these notebooks in modularized as `run_data_pipeline` in `src\LSTM\preprocess.py` performing Preprocessing -> Feature Engineering -> Saving Data**

---

## Modeling & Advanced Architectures
We conducted a comparison between classical time-series baselines and complex deep learning frameworks. All experiments were tracked using **Weights & Biases (W&B)**.

**1. Classical Baselines (`src/ARIMA/`, `src/ETS/`, `src/Prophet/`):**
* Exponential Smoothing (ETS)
* Prophet (with seasonal regressors)
* SARIMA & SARIMAX (incorporating exogenous weather variables)

**2. Deep Learning Architectures (`src/LSTM/`, `src/TFT/`):**
* LSTM + Dense Layer
* **Attention-LSTM**
* Encoder/Decoder LSTM (Hyperparameter tuned using Optuna)
* Temporal Fusion Transformer (TFT)

---

## Model Interpretability (`notebooks/03_interpretability.ipynb`)
To ensure the model is reliable for grid operators, we conducted a rigorous interpretability analysis. The analysis is lightweight for classical models and more detailed for the neural and hybrid models, where SHAP, attention, and counterfactual analysis are more informative.

---

## Repository Structure
The codebase follows strict software engineering practices, separating research notebooks from the production source code. Simplified repository structure:

```text
ML_FOR_DATA_ANALYTICS_PROJECT/
│
├── api_dashboard/           # Deployment: FastAPI Backend & Streamlit Frontend
├── data/                    # Raw, cleaned, and processed datasets
├── notebooks/               
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_interpretability.ipynb
│   ├── 04_final_report.ipynb
│   └── wandb_grouping.ipynb
├── src/                     # Modularized source code
│   ├── ARIMA/ ETS/ LSTM/ Prophet/ TFT/
|    ├── data_loader.py
│   ├── data_pipeline.py
│   ├── metrics.py
│   ├── wandb_setup.py
|   └── wandb_utils.py
├── tests/                   # Unit tests (pytest)
├── .github/workflows/       # CI/CD Pipeline definitions (ci.yml)
├── pyproject.toml           # Environment & formatting rules
├── requirements.txt         # Pinned dependencies
└── Dockerfile               # Containerization for the API
```

---

## Live Dashboard (`api_dashboard/`)
As an extension of the modeling phase, one of well-performing models (Attention-LSTM) is deployed with a live API and a dynamic streaming dashboard.

1. **The Backend (FastAPI):** Loads the trained PyTorch `.pth` weights and the `scaler.joblib` into memory upon startup. It listens for incoming HTTP POST requests containing scaled sequences of data, runs the forward pass, and returns the inverse-scaled predictions.
2. **The Frontend (Streamlit):** It streams raw CSV test data, pushes it through the exact preprocessing pipeline used in training, and polls the FastAPI endpoint. It visualizes the results using dynamic line charts and a custom power-flow diagram.

### Weights & Biases (W&B) Integration
We use the trained model logged into W&B Artifacts:
* During training, the exact scaler object and PyTorch weights (`model.pth`) are versioned and logged to W&B.
* In production, the backend and frontend download and initialize these exact artifacts. This ensures that the incoming production data is scaled using the exact parameters the model learned during training.

### Basic Monitoring Strategy
To ensure the model does not silently degrade in production, the Streamlit dashboard includes a **Model Monitoring**. 
* The UI calculates the **Mean Absolute Error (MAE)** over a sliding 20-step window.
* If the MAE crosses a predefined threshold, it triggers a visual **Data Drift Warning**, alerting the user that the incoming data distribution may have shifted away from the training data.

___ 
___
## Developer's Guide
### Quality Assurance Pipeline
After new commits remember to run:
```bash
pip install -r requirements.txt
```

We implement quality assurance for code formatting ( PEP 8 ) and code quality (over 900 lint rules) with black and ruff for folders `src/` `tests/` `notebooks/`. It's recommended to run:
```bash
black src/ tests/ notebooks/
ruff check src/ tests/ notebooks/
```
locally before pushing code to repo. 

- **black:** if you run it without `--check` flag it will automatically fix formatting in the files. With `--check` it only checks if reformatting is needed (this is part of pipeline now - it checks if you used it locally to reformat everything).
- **ruff:** you can add `--fix` flag and it will fix safe errors - for the rest simply read the error message and fix accordingly. **Important**: don't use star imports (`from ... import *`) as it's not considered good practice and will be always flagged by ruff. In special, justified cases you can add `# noqa` at the end of line to be ignored by ruff but use it only if necessary.

You can check what happens in CI pipeline in detail in `.github/workflows/ci.yml` file. After you create a Pull Request you can check if the pipeline is passed and see details in the Actions tab. If it failed, simply fix flagged errors and push again. Until the pipeline is passed you cannot merge with main.

---

### Weights & Biases setup

**Step 1 – Create account:** Go to https://wandb.ai. Make sure to use your student account and apply for Academic plan. Generate your API key.

**Step 2 – Use shared project:** Test your setup by running `src/wandb_setup.py`. You will be prompted to login - paste your API key.

Before you initiate wandb in your scripts, add:
```python
import wandb
wandb.login()

run = wandb.init(
    entity="ml-for-data-analytics-project",
    project="energy-forecasting",
    name="<model name>_<optional important param>", 
    config={"network_arch": "LSTM"} 
)
```

**Logging metrics (for ANY model):**
Please keep **exactly the same names** to log metrics so we can easily compare them in the runs table. You can use the functions from `src/metrics.py`.
```python
run.log({
    "val/mse": val_mse, 
    "val/rmse": rmse, 
    "val/mae": mae 
})
run.finish()
```
View results: https://wandb.ai/ml-for-data-analytics-project/energy-forecasting
</details>

---

### LSTM Framework (`src/LSTM/`)

**Structure:**
```text
src/LSTM/
│
├── preprocess.py        # data loading, encoding, lag features, full data pipeline
├── network_variant.py   # model definitions
├── direct.py            # training loop + wandb logging
├── main_lstm.py         # config + run script
```

**Running training:**
Inside `main_lstm.py` make sure all the fields in `CONFIG` are correct and run the file:
```bash
python src/LSTM/main_lstm.py
```
It will load data, encode categorical features, add lagged features, scale data, train the chosen model, and log results to W&B.

**Adding new model:**
Add a new class in `network_variant.py` e.g:
```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fun = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fun(x[:, -1, :])
```
In `main_lstm.py` import your model and configure `CONFIG`:
```python
from network_variant import MyModel

CONFIG = {
    "model": {
        "network_arch": MyModel,
        "network_params": {
            "hidden_size": 64
        }
    }
}
```