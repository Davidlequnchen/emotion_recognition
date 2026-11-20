"""Check the configured device with real matmul/Conv1d/backward, without models.

Run ``python emotion2vec_adaptation/utilities/check_environment.py``. Device and precision
come from config/common.yaml; reports use config/environment_check.yaml. A CPU
pass is a CPU check, never evidence of a B200 pass. CUDA requests fail explicitly
when a CUDA GPU or the correct PyTorch build is unavailable.
"""
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from utilities.config import PROJECT_DIR, resolve_path, validate_settings
from utilities.hardware import device_environment, validate_device
from utilities.runtime import initialize_runtime, write_json


def check_environment() -> Path:
    common = yaml.safe_load((PROJECT_DIR / "config/common.yaml").read_text(encoding="utf-8"))
    output_cfg = yaml.safe_load((PROJECT_DIR / "config/environment_check.yaml").read_text(encoding="utf-8"))
    if not isinstance(output_cfg, dict) or set(output_cfg) != {"environment_check"} or set(output_cfg["environment_check"]) != {"output_root"}:
        raise ValueError("environment_check.yaml must contain environment_check.output_root")
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%dT%H%M%S")
    run = resolve_path(output_cfg["environment_check"]["output_root"]) / f"{stamp}_{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "requested_runtime": common["runtime"], "environment": device_environment(),
              "scope": "Tiny real device kernels only; full model validation must run separately on the target GPU."}
    try:
        validate_settings(common)
        device = initialize_runtime(common["runtime"])
        report["environment"] = device_environment(device)
        report["device_checks"] = validate_device(device, common["runtime"].get("precision", "auto"), run_kernels=True)
        report["status"] = "passed"
        report["b200_hardware_checked"] = device.type == "cuda" and "B200" in report["environment"]["selected_cuda_device"]["name"].upper()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["b200_hardware_checked"] = False
        raise
    finally:
        write_json(run / "environment_report.json", report)
        print(f"Environment report: {run / 'environment_report.json'}", flush=True)
    print(f"Device check passed: {device}; precision={report['device_checks']['resolved_precision']}", flush=True)
    return run


if __name__ == "__main__":
    check_environment()
