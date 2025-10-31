"""
Minimal MCP server exposing `list_mice` using DatasetAPI.

Run quick check:
    python -m src.tools.mcp_server --dataset-root data --once

Start server (stdio):
    python -m src.tools.mcp_server
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context  # type: ignore

from src.tools.dataset.dataset_api import DatasetAPI  # reuse repo API

app = FastMCP("neural-agent-dataset", "Dataset tools over MCP")


@app.tool()
def list_mice(dataset_root: str, include_full_paths: bool = True) -> Dict[str, List[Dict[str, str]]]:
    """List mice (subjects) using DatasetAPI.

    Returns: {"mice": [{"name": <subject_id>, "path": <path>}, ...]}
    """
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dataset root not found or not a directory: {root}")

    api = DatasetAPI(root)
    subject_ids = api.list_subjects()

    mice: List[Dict[str, str]] = []
    for sid in subject_ids:
        subject_dir = root / f"Subject-{sid}"
        mice.append({
            "name": sid,
            "path": str(subject_dir.resolve() if include_full_paths else subject_dir),
        })
    return {"mice": mice}


@app.tool()
def ping() -> str:
    return "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        print(json.dumps(list_mice(args.dataset_root), indent=2))
    else:
        app.run()