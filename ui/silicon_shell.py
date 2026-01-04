# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PyQt5 import QtWidgets


@dataclass(frozen=True)
class SiliconLayoutSpec:
    """SiliconUI 页面布局规范。

    - page: SiPage 页面容器
    - attachment: 主体区域（承载预览区、控制区、参数区、状态栏）
    """

    page_title: str = "相机 / 视觉"
    page_padding: int = 16
    sidebar_hint: str = "Camera"
    window_title: str = "Camera (SiliconUI)"


def _try_import_siui() -> Tuple[bool, Optional[tuple], Optional[Exception]]:
    try:
        from siui.templates.application.application import SiliconApplication  # type: ignore
        from siui.components.page import SiPage  # type: ignore
        from siui.core import SiGlobal  # type: ignore
        return True, (SiliconApplication, SiPage, SiGlobal), None
    except Exception as exc:  # noqa: BLE001
        return False, None, exc


def create_silicon_shell(content: QtWidgets.QWidget, spec: SiliconLayoutSpec = SiliconLayoutSpec()):
    have_siui, objs, err = _try_import_siui()
    if not have_siui:
        raise RuntimeError(f"未检测到 PyQt-SiliconUI: {err}")

    SiliconApplication, SiPage, SiGlobal = objs  # type: ignore

    class _CameraHostPage(SiPage):  # type: ignore
        def __init__(self, parent=None):
            super().__init__(parent)
            try:
                self.setPadding(spec.page_padding)
            except Exception:
                pass
            try:
                self.setTitle(spec.page_title)
            except Exception:
                pass

            content.setParent(self)
            try:
                self.setAttachment(content)
            except Exception:
                lay = QtWidgets.QVBoxLayout(self)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.addWidget(content)

    class _SiliconCameraApp(SiliconApplication):  # type: ignore
        def __init__(self):
            super().__init__()
            self.setWindowTitle(spec.window_title)
            try:
                self.layerMain().setTitle(spec.window_title)
            except Exception:
                pass

            page = _CameraHostPage(self)
            icon = None
            try:
                icon = SiGlobal.siui.iconpack.get("ic_fluent_home_filled")
            except Exception:
                icon = None

            try:
                self.layerMain().addPage(page, icon=icon, hint=spec.sidebar_hint, side="top")
                self.layerMain().setPage(0)
            except Exception:
                page.setParent(self)
                lay = QtWidgets.QVBoxLayout(self)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.addWidget(page)

            try:
                SiGlobal.siui.reloadAllWindowsStyleSheet()
            except Exception:
                pass

    return _SiliconCameraApp()
