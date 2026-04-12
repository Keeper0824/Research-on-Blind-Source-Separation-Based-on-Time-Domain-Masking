import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Model.train_bpsk_small_common import add_common_train_arguments, build_model, train_bpsk_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Baseline TasNet on the small BPSK-only communication dataset."
    )
    default_dataset_root = str(REPO_ROOT / "Dataset" / "CommBPSK_4v16")
    default_save_dir = str(REPO_ROOT / "checkpoints" / "BPSKSmall" / "baseline")
    return add_common_train_arguments(
        parser=parser,
        default_dataset_root=default_dataset_root,
        default_save_dir=default_save_dir,
    ).parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_bpsk_model(
        model=build_model("baseline"),
        model_name="baseline",
        args=args,
    )
