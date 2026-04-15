import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"


def run(command):
    print("Running:", " ".join(command))
    subprocess.check_call(command)


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


def main():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    status = torch_status()
    print("Final torch status:", status)
    if status["cuda_available"]:
        print(f"CUDA is available on: {status['device_name']}")
    else:
        print("CUDA is not available in the current Python environment. Restart the kernel after install, then verify again.")


if __name__ == "__main__":
    main()
