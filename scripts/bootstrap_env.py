import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCHAUDIO_VERSION = "2.8.0"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def run(command):
    print("Running:", " ".join(command))
    subprocess.check_call(command)


def has_nvidia_gpu():
    if os.environ.get("LLM_DEMO_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        return False

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except OSError:
        return False


def torch_status():
    try:
        import torch  # type: ignore

        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        return {
            "installed": False,
            "version": None,
            "cuda_build": None,
            "cuda_available": False,
            "device_name": None,
        }


def ensure_torch():
    wants_cuda = has_nvidia_gpu()
    status = torch_status()

    print("Current torch status:", status)
    print("NVIDIA GPU detected by bootstrap:", wants_cuda)

    torch_ok = status["installed"]
    if wants_cuda:
        torch_ok = torch_ok and status["cuda_build"] is not None
    else:
        torch_ok = torch_ok

    if torch_ok:
        print("PyTorch installation is already acceptable for this machine.")
        return

    run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    if wants_cuda:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"torch=={TORCH_VERSION}",
                f"torchvision=={TORCHVISION_VERSION}",
                f"torchaudio=={TORCHAUDIO_VERSION}",
                "--index-url",
                CUDA_INDEX_URL,
            ]
        )
    else:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"torch=={TORCH_VERSION}",
                f"torchvision=={TORCHVISION_VERSION}",
                f"torchaudio=={TORCHAUDIO_VERSION}",
                "--index-url",
                CPU_INDEX_URL,
            ]
        )


def main():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    ensure_torch()
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    final_status = torch_status()
    print("Final torch status:", final_status)
    if final_status["cuda_available"]:
        print(f"CUDA is available on: {final_status['device_name']}")
    else:
        print("CUDA is not available. The notebooks will run on CPU unless a CUDA-capable torch build is installed.")


if __name__ == "__main__":
    main()
