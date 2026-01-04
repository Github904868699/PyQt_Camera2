# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict

from PyQt5 import QtCore, QtGui, QtWidgets

from ui.callbacks import UiCallbacks


def setup_main_window_ui(
    main_window: QtWidgets.QMainWindow,
    callbacks: UiCallbacks,
    title: str = "Camera",
) -> None:
    main_window.setWindowTitle(title)
    main_window.resize(2020, 700)

    central = QtWidgets.QWidget()
    main_window.setCentralWidget(central)
    outer = QtWidgets.QVBoxLayout(central)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(8)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    outer.addWidget(splitter, 1)

    # 左侧
    left = QtWidgets.QWidget()
    left.setMinimumWidth(260)
    left.setMaximumWidth(360)
    splitter.addWidget(left)

    left_lay = QtWidgets.QVBoxLayout(left)
    left_lay.setContentsMargins(8, 8, 8, 8)
    left_lay.setSpacing(10)

    main_window.slot_widgets = {}

    for slot in (1, 2):
        gb_slot = QtWidgets.QGroupBox(f"摄像头{slot}")
        left_lay.addWidget(gb_slot)
        form = QtWidgets.QFormLayout(gb_slot)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)

        source_combo = QtWidgets.QComboBox()
        source_combo.addItem("海康网络相机", "hik")
        source_combo.addItem("USB", "usb")
        idx = source_combo.findData(main_window.slot_sources.get(slot, "hik"))
        if idx >= 0:
            source_combo.setCurrentIndex(idx)
        source_combo.currentIndexChanged.connect(lambda i, s=slot: callbacks.on_slot_source_changed(s, i))
        form.addRow("类型：", source_combo)

        hik_row = QtWidgets.QHBoxLayout()
        hik_combo = QtWidgets.QComboBox()
        btn_hik_refresh = QtWidgets.QPushButton("刷新")
        btn_hik_refresh.setFixedWidth(70)
        btn_hik_refresh.clicked.connect(callbacks.on_refresh_hik)
        hik_row.addWidget(hik_combo, 1)
        hik_row.addWidget(btn_hik_refresh, 0)
        hik_wrap = QtWidgets.QWidget()
        hik_wrap.setLayout(hik_row)

        usb_row = QtWidgets.QHBoxLayout()
        usb_combo = QtWidgets.QComboBox()
        btn_usb_refresh = QtWidgets.QPushButton("刷新")
        btn_usb_refresh.setFixedWidth(70)
        btn_usb_refresh.clicked.connect(callbacks.on_refresh_usb)
        usb_row.addWidget(usb_combo, 1)
        usb_row.addWidget(btn_usb_refresh, 0)
        usb_wrap = QtWidgets.QWidget()
        usb_wrap.setLayout(usb_row)

        idx_stack = QtWidgets.QStackedWidget()
        idx_stack.setContentsMargins(0, 0, 0, 0)
        idx_stack.addWidget(hik_wrap)
        idx_stack.addWidget(usb_wrap)
        form.addRow("索引：", idx_stack)

        ctrl_row = QtWidgets.QHBoxLayout()
        btn_reopen = QtWidgets.QPushButton("重连")
        btn_stop = QtWidgets.QPushButton("停止")
        btn_auto_adjust = QtWidgets.QPushButton("调节")
        btn_reopen.clicked.connect(lambda _=None, s=slot: callbacks.on_reopen_camera(s))
        btn_stop.clicked.connect(lambda _=None, s=slot: callbacks.on_stop_camera(s))
        btn_auto_adjust.clicked.connect(lambda _=None, s=slot: callbacks.on_auto_adjust(s))
        ctrl_row.addWidget(btn_reopen, 1)
        ctrl_row.addWidget(btn_stop, 1)
        ctrl_row.addWidget(btn_auto_adjust, 1)
        ctrl_wrap = QtWidgets.QWidget()
        ctrl_wrap.setLayout(ctrl_row)
        form.addRow("控制：", ctrl_wrap)

        chk_yolo = QtWidgets.QCheckBox("启用 YOLO")
        chk_yolo.setChecked(main_window.slot_yolo_enabled.get(slot, False))
        chk_yolo.stateChanged.connect(lambda state, s=slot: callbacks.on_yolo_enabled_changed(s, state))
        form.addRow("YOLO：", chk_yolo)

        combo_model = QtWidgets.QComboBox()
        for m in main_window.models:
            combo_model.addItem(os.path.basename(m), m)
        if not main_window.models:
            combo_model.setEditable(True)
            combo_model.lineEdit().setPlaceholderText("请选择模型文件")
        idx_model = combo_model.findData(main_window.slot_model_paths.get(slot, main_window.default_model))
        if idx_model >= 0:
            combo_model.setCurrentIndex(idx_model)
        combo_model.currentIndexChanged.connect(lambda i, s=slot: callbacks.on_yolo_model_changed(s, i))
        form.addRow("模型：", combo_model)

        main_window.slot_widgets[slot] = {
            "source": source_combo,
            "hik_combo": hik_combo,
            "usb_combo": usb_combo,
            "hik_refresh": btn_hik_refresh,
            "usb_refresh": btn_usb_refresh,
            "idx_stack": idx_stack,
            "btn_reopen": btn_reopen,
            "btn_stop": btn_stop,
            "btn_auto": btn_auto_adjust,
            "chk_yolo": chk_yolo,
            "combo_model": combo_model,
        }

    # YOLO 全局设置
    gb_yolo = QtWidgets.QGroupBox("YOLO 设置")
    left_lay.addWidget(gb_yolo)
    yolo_form = QtWidgets.QFormLayout(gb_yolo)
    yolo_form.setContentsMargins(10, 10, 10, 10)
    yolo_form.setSpacing(8)

    main_window.spin_conf = QtWidgets.QDoubleSpinBox()
    main_window.spin_conf.setDecimals(2)
    main_window.spin_conf.setRange(0.0, 1.0)
    main_window.spin_conf.setSingleStep(0.01)
    main_window.spin_conf.setValue(main_window.yolo_conf)
    main_window.spin_conf.valueChanged.connect(callbacks.on_yolo_conf_changed)
    yolo_form.addRow("置信度：", main_window.spin_conf)

    main_window.combo_style = QtWidgets.QComboBox()
    main_window.combo_style.addItem("经典绿", 0)
    main_window.combo_style.addItem("相机红点", 1)
    main_window.combo_style.addItem("YOLO", 2)
    main_window.combo_style.addItem("配色 A", 3)
    main_window.combo_style.addItem("配色 B", 4)
    main_window.combo_style.addItem("极简黑白", 5)
    main_window.combo_style.addItem("霓虹", 6)
    main_window.combo_style.addItem("半透明", 7)
    main_window.combo_style.setCurrentIndex(main_window.yolo_style)
    main_window.combo_style.currentIndexChanged.connect(callbacks.on_yolo_style_changed)
    yolo_form.addRow("框样式：", main_window.combo_style)

    # Modbus 分组
    gb_modbus = QtWidgets.QGroupBox("Modbus")
    left_lay.addWidget(gb_modbus)
    modbus_form = QtWidgets.QFormLayout(gb_modbus)
    modbus_form.setContentsMargins(10, 10, 10, 10)
    modbus_form.setSpacing(8)

    main_window.lbl_modbus_status = QtWidgets.QLabel("—")
    main_window.lbl_modbus_reg0 = QtWidgets.QLabel("—")
    main_window.lbl_modbus_reg01 = QtWidgets.QLabel("—")
    main_window.lbl_modbus_reg1 = QtWidgets.QLabel("—")
    main_window.lbl_modbus_reg2 = QtWidgets.QLabel("—")
    modbus_form.addRow("状态：", main_window.lbl_modbus_status)
    modbus_form.addRow("摄像头1-寄存器0：", main_window.lbl_modbus_reg0)
    modbus_form.addRow("摄像头2-寄存器1：", main_window.lbl_modbus_reg01)
    modbus_form.addRow("摄像头1-寄存器2：", main_window.lbl_modbus_reg1)
    modbus_form.addRow("摄像头2-寄存器3：", main_window.lbl_modbus_reg2)

    left_lay.addStretch(1)

    # 右侧
    right = QtWidgets.QWidget()
    splitter.addWidget(right)
    right_lay = QtWidgets.QVBoxLayout(right)
    right_lay.setContentsMargins(8, 8, 8, 8)
    right_lay.setSpacing(8)

    header_row = QtWidgets.QHBoxLayout()
    main_window.lbl_cam1 = QtWidgets.QLabel("摄像头1: —")
    main_window.lbl_fps1 = QtWidgets.QLabel("FPS1: —")
    main_window.lbl_cam2 = QtWidgets.QLabel("摄像头2: —")
    main_window.lbl_fps2 = QtWidgets.QLabel("FPS2: —")

    cam1_row = QtWidgets.QHBoxLayout()
    cam1_row.addWidget(main_window.lbl_cam1)
    cam1_row.addStretch(1)
    cam1_row.addWidget(main_window.lbl_fps1)

    cam2_row = QtWidgets.QHBoxLayout()
    cam2_row.addWidget(main_window.lbl_cam2)
    cam2_row.addStretch(1)
    cam2_row.addWidget(main_window.lbl_fps2)

    header_row.addLayout(cam1_row, 1)
    header_row.addSpacing(12)
    header_row.addLayout(cam2_row, 1)
    right_lay.addLayout(header_row)

    video_row = QtWidgets.QHBoxLayout()
    main_window.video_lbl1 = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
    main_window.video_lbl1.setMinimumSize(480, 360)
    main_window.video_lbl1.setFrameShape(QtWidgets.QFrame.StyledPanel)
    main_window.video_lbl1.setAutoFillBackground(True)

    main_window.video_lbl2 = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
    main_window.video_lbl2.setMinimumSize(480, 360)
    main_window.video_lbl2.setFrameShape(QtWidgets.QFrame.StyledPanel)
    main_window.video_lbl2.setAutoFillBackground(True)

    pal = main_window.video_lbl1.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
    main_window.video_lbl1.setPalette(pal)
    main_window.video_lbl2.setPalette(pal)

    video_row.addWidget(main_window.video_lbl1, 1)
    video_row.addWidget(main_window.video_lbl2, 1)
    right_lay.addLayout(video_row, 1)

    main_window.status_label = QtWidgets.QLabel("就绪")
    main_window.status_label.setMinimumHeight(22)
    main_window.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    right_lay.addWidget(main_window.status_label, 0)

    main_window.toast_timer = QtCore.QTimer(main_window)
    main_window.toast_timer.setSingleShot(True)
    main_window.toast_timer.timeout.connect(lambda: main_window.status_label.setText(""))

    main_window.modbus_ui_timer = QtCore.QTimer(main_window)
    main_window.modbus_ui_timer.setInterval(800)
    main_window.modbus_ui_timer.timeout.connect(main_window._refresh_modbus_status)
    main_window.modbus_ui_timer.start()
    main_window._refresh_modbus_status()

    main_window.watchdog_timer = QtCore.QTimer(main_window)
    main_window.watchdog_timer.setInterval(3000)
    main_window.watchdog_timer.timeout.connect(main_window._watchdog_tick)
    main_window.watchdog_timer.start()

    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([300, 860])
