"""Edit config/common.yaml and config/head.yaml, then run python train_head.py."""
from utilities.config import load_settings
from adaptation.head import HeadTrainer

if __name__ == "__main__":
    HeadTrainer(load_settings("head")).run()
