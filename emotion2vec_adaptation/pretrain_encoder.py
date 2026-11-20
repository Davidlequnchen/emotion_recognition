"""Edit config/common.yaml and config/pretrain.yaml; no emotion labels required."""
from utilities.config import load_settings
from adaptation.pretrain import ContinualPretrainer

if __name__ == "__main__":
    ContinualPretrainer(load_settings("pretrain")).run()
