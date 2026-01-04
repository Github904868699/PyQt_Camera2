# -*- coding: utf-8 -*-
"""
Camera_SiliconUI_Wrapper.py

用途：
- 使用 SiliconUI 作为窗口外壳。
- 复用 Camera.py 的主窗口作为内容区域，并挂载到 SiliconUI 的页面容器。
"""

import sys

from PyQt5 import QtWidgets

from ui.silicon_shell import SiliconLayoutSpec, create_silicon_shell


def main():
    app = QtWidgets.QApplication(sys.argv)

    try:
        from Camera import MainWindow as CameraMainWindow  # noqa: WPS433
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "导入失败", f"无法导入 Camera.py：\n{exc}")
        return 1

    camera_win = CameraMainWindow()

    try:
        shell = create_silicon_shell(camera_win.takeCentralWidget() or camera_win, SiliconLayoutSpec())
    except Exception as exc:
        QtWidgets.QMessageBox.information(
            camera_win,
            "未检测到 PyQt-SiliconUI",
            "当前未安装 PyQt-SiliconUI，将以普通窗口方式启动。\n\n"
            f"导入错误：{exc}",
        )
        camera_win.show()
        return app.exec_()

    shell.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
