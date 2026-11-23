"""Local hardware observations. Torch is optional on CPU-only installations."""

import shutil
import subprocess


def telemetry():
    info = {"device": "cpu", "cuda_available": False}
    command = shutil.which("nvidia-smi")
    if command:
        try:
            result = subprocess.run(
                [
                    command,
                    "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            values = result.stdout.strip().splitlines()[0].split(", ")
            info.update(
                zip(
                    [
                        "gpu_name",
                        "driver",
                        "capability",
                        "vram_mib",
                        "used_mib",
                        "utilization_percent",
                    ],
                    values,
                )
            )
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            info["telemetry_error"] = str(exc)
    try:
        import torch

        info.update(
            torch_version=str(torch.__version__),
            cuda_available=torch.cuda.is_available(),
            cuda_build=torch.version.cuda,
        )
        if info["cuda_available"]:
            info.update(
                device="cuda",
                allocated_mib=round(torch.cuda.memory_allocated() / 1048576, 2),
            )
    except (ImportError, OSError) as exc:
        info["torch_error"] = str(exc)
    return info


def verify():
    info = telemetry()
    if "torch_version" in info:
        import torch

        try:
            x = torch.arange(4, device=info["device"], dtype=torch.float32).reshape(2, 2)
            info["tensor_result"] = (x @ x.T).cpu().tolist()
            info["verified"] = info["tensor_result"] == [[1, 3], [3, 13]]
        except RuntimeError as exc:
            info.update(verified=False, execution_error=str(exc))
    else:
        info["verified"] = False
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(verify(), indent=2))
