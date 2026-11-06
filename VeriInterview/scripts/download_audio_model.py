"""Explicit one-time download; audio inference itself defaults to offline."""
import argparse
from pathlib import Path
from faster_whisper.utils import download_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base")
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parents[1] / "storage" / "models"))
    args = parser.parse_args()
    print(download_model(args.model, cache_dir=args.cache_dir))
