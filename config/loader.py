# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

CONFIG_PATH = "config.json"


def resource_path(rel: str) -> str:
    """打包后获取资源路径（图标等资源）"""
    base = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)
    return str(Path(base, rel))


def safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_config(path: str = CONFIG_PATH) -> Dict:
    cfg = None
    search_paths = []
    if path:
        search_paths.append(path)
        res_path = resource_path(path)
        if res_path != path:
            search_paths.append(res_path)

    for candidate in search_paths:
        cfg = safe_load_json(candidate, default=None)
        if cfg:
            break

    if not cfg:
        cfg = {
            "server": {"host": "0.0.0.0", "port": 502},
            "class_map": {},
        }
    cfg.setdefault("server", {"host": "0.0.0.0", "port": 502})
    cfg.setdefault("class_map", {})
    return cfg
