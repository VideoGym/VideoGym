"""Download public VBVR-Pro Wan releases."""

import argparse
import json
from pathlib import Path

RELEASES = {
    "5b-rlvr": "Video-Reason/VBVR-Pro-Wan2.2-TI2V-5B-RLVR",
    "a14b": "Video-Reason/VBVR-Pro-Wan2.2-I2V-A14B",
}


def prepare(output, model="5b-rlvr"):
    from huggingface_hub import snapshot_download

    output = Path(output).resolve()
    repository = RELEASES[model]
    snapshot_download(repository, local_dir=output, max_workers=3)
    if model == "5b-rlvr" and not (output / "pipeline.py").is_file():
        raise ValueError("the RLVR release requires its bundled custom pipeline")
    return {
        "model": model,
        "repository": repository,
        "path": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=RELEASES, default="5b-rlvr")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.model)))


if __name__ == "__main__":
    main()
