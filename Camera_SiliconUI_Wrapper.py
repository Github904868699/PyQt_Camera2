# -*- coding: utf-8 -*-
"""
Camera_SiliconUI_Wrapper.py

目的：
- 在不大改你现有业务逻辑（相机/YOLO/Modbus/TCP 等）的情况下，
  把窗口“外壳”换成 PyQt‑SiliconUI 的 SiliconApplication（背景、标题栏、整体质感一致）。
- 内部核心界面直接复用 Camera_SiliconLike.py 的 MainWindow 中的 centralWidget，
  所以你现有按钮/下拉框/框框样式（QSS）也能继续保持。

用法：
1) 确保与你的 Camera_SiliconLike.py 放在同一目录
2) 安装 PyQt‑SiliconUI（见聊天里的 uv 命令）
3) 运行：
   python Camera_SiliconUI_Wrapper.py
"""

import sys

from PyQt5 import QtCore, QtWidgets


def _try_import_siui():
    """延迟导入，避免未安装时直接崩。"""
    try:
        from siui.templates.application.application import SiliconApplication  # type: ignore
        from siui.components.page import SiPage  # type: ignore
        from siui.core import SiGlobal  # type: ignore
        return True, (SiliconApplication, SiPage, SiGlobal), None
    except Exception as exc:  # noqa
        return False, None, exc


class CameraHostPage:  # 动态继承 SiPage，避免无 siui 时解析失败
    pass


class SiliconCameraApp:  # 动态继承 SiliconApplication
    pass


def main():
    have_siui, objs, err = _try_import_siui()

    app = QtWidgets.QApplication(sys.argv)

    # 你的界面逻辑（含 QSS 皮肤）在这里
    try:
        from Camera_SiliconLike import MainWindow as CameraMainWindow  # noqa
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "导入失败", f"无法导入 Camera_SiliconLike.py：\n{exc}")
        return 1

    camera_win = CameraMainWindow()

    if not have_siui:
        # 没装 siui：就按原样启动（依旧是 SiliconLike 皮肤）
        QtWidgets.QMessageBox.information(
            camera_win,
            "未检测到 PyQt-SiliconUI",
            "当前未安装 PyQt-SiliconUI，将以普通窗口方式启动。\n\n"
            f"导入错误：{err}"
        )
        camera_win.show()
        return app.exec_()

    SiliconApplication, SiPage, SiGlobal = objs  # type: ignore

    # 动态创建 Page 与 App 的子类（避免类型检查问题）
    class _CameraHostPage(SiPage):  # type: ignore
        def __init__(self, parent=None):
            super().__init__(parent)
            # 页面边距：让你的 centralWidget 自己控制内部布局即可
            try:
                self.setPadding(16)
            except Exception:
                pass
            try:
                self.setTitle("相机 / 视觉")
            except Exception:
                pass

            # 复用原 MainWindow 的 centralWidget 作为页面主体
            central = camera_win.takeCentralWidget()
            if central is None:
                # 极少数情况下 takeCentralWidget 取不到，就退化为直接用整个窗口
                central = camera_win
            central.setParent(self)

            # SiliconUI 的 Page 通常用 setAttachment 注入主体
            try:
                self.setAttachment(central)
            except Exception:
                # 兜底：直接放到一个布局里
                lay = QtWidgets.QVBoxLayout(self)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.addWidget(central)

    class _SiliconCameraApp(SiliconApplication):  # type: ignore
        def __init__(self):
            super().__init__()

            # 标题
            self.setWindowTitle("Camera (SiliconUI)")
            try:
                self.layerMain().setTitle("Camera (SiliconUI)")
            except Exception:
                pass

            # 页面
            page = _CameraHostPage(self)

            # 侧边栏图标：用一个已知存在的（home）作兜底
            icon = None
            try:
                icon = SiGlobal.siui.iconpack.get("ic_fluent_home_filled")
            except Exception:
                icon = None

            try:
                self.layerMain().addPage(page, icon=icon, hint="Camera", side="top")
                self.layerMain().setPage(0)
            except Exception:
                # 如果模板 API 变化，至少把 page 显示出来
                page.setParent(self)
                lay = QtWidgets.QVBoxLayout(self)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.addWidget(page)

            # 关键：刷新 SiliconUI 全局样式
            try:
                SiGlobal.siui.reloadAllWindowsStyleSheet()
            except Exception:
                pass

        def closeEvent(self, event):
            # 关闭时把你的采集线程/连接也一起关掉
            try:
                camera_win.close()
            except Exception:
                pass
            super().closeEvent(event)

    win = _SiliconCameraApp()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
