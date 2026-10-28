# models/

One sub-folder per model. Model files are large, so git ignores them (only the empty
folder placeholders are tracked). Download them on each machine with:

```bash
python scripts/download_model.py   # downloads the model named by `model:` in config/config.yaml
```

| Folder | Source | Status |
|---|---|---|
| `emotion2vec_plus_large/` | [HuggingFace](https://huggingface.co/emotion2vec/emotion2vec_plus_large) · [ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) | in use |
| *(next model)* | | placeholder |

## emotion2vec_plus_large

| File | Content |
|---|---|
| `model.pt` | weights, 1.9 GB: 162 M fp32 weights plus the optimizer state saved during training |
| `config.yaml` | network architecture, read by FunASR to build the model |
| `configuration.json` | tells FunASR which file is the config, the weights and the label list |
| `tokens.txt` | the 9 output classes, in output order |
| `README.md`, `example/test.wav` | model card (license) and an example clip |

The official release has no `config.json`; `configuration.json` plays that role.

## Adding a model

1. Register it under `models:` in `config/config.yaml` (`dir`, `hf_repo`, `ms_repo`).
2. Make it active: `model: <name>` in the config.
3. Download it: `python scripts/download_model.py`.

The other emotion2vec+ sizes (`emotion2vec_plus_base`, `emotion2vec_plus_seed`) use the same
FunASR code, so they should only need steps 1–3. Other model families will need their own loading code.
