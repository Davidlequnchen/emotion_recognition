"""Edit config/predict.yaml, then run python predict.py."""
from utilities.config import load_settings
from adaptation.inference import run_prediction

if __name__ == "__main__":
    run_prediction(load_settings("predict"))
