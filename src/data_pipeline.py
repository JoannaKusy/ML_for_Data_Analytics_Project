from LSTM.preprocess import run_data_pipeline

RAW_DIR = "./data/raw"
PIPELINE_OUT_DIR = "./data/processed"

print("Running full pipeline...")
run_data_pipeline(RAW_DIR, PIPELINE_OUT_DIR)
print("Saved in ", PIPELINE_OUT_DIR)
