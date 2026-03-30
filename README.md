# Machine-Learning-for-Data-Analytics-Project

After new commits remeber to run:


```bash
pip install -r requirements.txt
```

---

# Detailed intructions

---

<details>
<summary><b>Weights & Biases (W&B) setup</b></summary>

### Step 1 – Create account
Go to: https://wandb.ai
Make sure to use your student account and apply for Academic plan.
### Step 2 – Login
Generate your API key on the website and save it. You will need it to login when using wandb in this project.

### Step 3 – Use shared project
As a first step test your setup by running `src\wandb_setup.py` - you can change `name` field before if you want. You will be prompted to login - paste your API key generated earlier.
Then go to https://wandb.ai/ml-for-data-analytics-project/test_project to check if you can see your new run there.

If you want to add wandb tracking to other models (outside of LSTM directory - you can see how it's set up there in direct.py file) follow these guidelines.

Before you initiate wandb add a line 
```bash
wandb.login()
```
and paste your API key when prompted (if you logged in earlier it should use your profile automatically)

Then **always** use 
- **Entity:** `ml-for-data-analytics-project`
- **Project:** `energy-forecasting`
to log everything in our project. 

We want to have a consistent structure of logged metrics and names in the project so please follow this structure when adding wandb to you files:

```python
import wandb

wandb.login()

run = wandb.init(
    entity="ml-for-data-analytics-project",
    project="energy-forecasting",
    name="<model name>_<optional important param>", #how your run will be displayed - you can also edit it on the website later
    config={"network_arch": "LSTM"} #optional field for parameters etc you want to log (you don't have to provide config argument at all)
)
```
---

### Logging metrics (for ANY model: LSTM, ARIMA, etc.)
Please keep **exactly the same names** as here to log metrics so we can easily compare them in the runs table.
You can also add some additional ones if you want, but I suggest to have these ones common for all models. 
You can use the functions from `src/metrics.py` file for the ones below.
```python
run.log({
    "val/mse": val_mse, 
    "val/rmse": rmse, 
    "val/mae": mae 
})
```

Finish run:

```python
run.finish()
```
---

### Viewing results

Go to 
https://wandb.ai/ml-for-data-analytics-project/energy-forecasting

If you don't have access please contact Karolina.

</details>

---

<details>
<summary><b> LSTM framework</b></summary>

### structure

```
src/LSTM/
│
├── preprocess.py        # data loading, encoding, lag features
├── network_variant.py   # model definitions
├── direct.py            # training loop + wandb logging
├── main_lstm.py         # config + run script
```

### Running training

Inside `main_lstm.py` make sure all the fields in `CONFIG` are correct and run the file

```bash
python src/LSTM/main_lstm.py
```
It will

1. Load data  
2. Encode categorical features  
3. Add lagged features  
4. Scale data  
5. Train chosen model  
6. Log results to W&B 
---
### Adding new model

Add a new class in `network_variant.py` e.g:

```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fun = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fun(x[:, -1, :])
```

---


In `main_lstm.py` import your model and configure `CONFIG` 

```python
from network_variant import  MyModel

CONFIG = {
    "model": {
        "network_arch": MyModel,
        "network_params": {
            "hidden_size": 64
        }
    }
}
```
After editing  `CONFIG` accordingly run 
```bash
python src/LSTM/main_lstm.py
```
</details>