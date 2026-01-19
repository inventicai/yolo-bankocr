# src/main.py
import argparse
import yaml
import asyncio
from pathlib import Path
from src.inference.runner import run_folder

def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO OCR pipeline on a folder of images/PDFs")
    p.add_argument(
        "--input_dir",
        default=r"C:\Users\Dudes\Desktop\RegressiveTest",
        help="Folder with images or pdfs (default: data/raw)"
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)"
    )
    return p.parse_args()

async def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = yaml.safe_load(open(cfg_path, "r"))

    input_dir = Path(args.input_dir)
    # create data/raw if it doesn't exist so users get a clear folder to populate
    input_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Using input directory: {input_dir.resolve()}")
    print(f"[+] Using config file: {cfg_path.resolve()}")

    # Await the async runner
    await run_folder(input_dir, cfg)


if __name__ == "__main__":
    # Correctly run the async main function
    asyncio.run(main())