# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class UiCallbacks:
    on_slot_source_changed: Callable[[int, int], None]
    on_refresh_hik: Callable[[], None]
    on_refresh_usb: Callable[[], None]
    on_hik_index_changed: Callable[[int, int], None]
    on_usb_index_changed: Callable[[int, int], None]
    on_reopen_camera: Callable[[int], None]
    on_stop_camera: Callable[[int], None]
    on_auto_adjust: Callable[[int], None]
    on_yolo_enabled_changed: Callable[[int, int], None]
    on_yolo_model_changed: Callable[[int, int], None]
    on_yolo_conf_changed: Callable[[float], None]
    on_yolo_style_changed: Callable[[int], None]
