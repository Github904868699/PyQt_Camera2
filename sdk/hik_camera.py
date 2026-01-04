# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import threading
from typing import Optional

_MV_CAMERA_SPEC = importlib.util.find_spec("MvCameraControl_class")
_CAMERA_PARAMS_SPEC = importlib.util.find_spec("CameraParams_header")
_CAMERA_CONST_SPEC = importlib.util.find_spec("CameraParams_const")
_PIXEL_TYPE_SPEC = importlib.util.find_spec("PixelType_header")
_MV_ERROR_SPEC = importlib.util.find_spec("MvErrorDefine_const")

if None in (
    _MV_CAMERA_SPEC,
    _CAMERA_PARAMS_SPEC,
    _CAMERA_CONST_SPEC,
    _PIXEL_TYPE_SPEC,
    _MV_ERROR_SPEC,
):
    MvCamera = None  # type: ignore
    MV_CC_DEVICE_INFO = None  # type: ignore
    MV_CC_DEVICE_INFO_LIST = None  # type: ignore
    MV_FRAME_OUT_INFO_EX = None  # type: ignore
    MVCC_INTVALUE = None  # type: ignore
    MV_CC_PIXEL_CONVERT_PARAM = None  # type: ignore
    MV_GIGE_DEVICE = 0  # type: ignore
    MV_ACCESS_Exclusive = 1  # type: ignore
    PixelType_Gvsp_BGR8_Packed = 0  # type: ignore
    PixelType_Gvsp_RGB8_Packed = 0  # type: ignore
    PixelType_Gvsp_Mono8 = 0  # type: ignore
    PixelType_Gvsp_BayerRG8 = 0  # type: ignore
    PixelType_Gvsp_BayerBG8 = 0  # type: ignore
    PixelType_Gvsp_BayerGB8 = 0  # type: ignore
    PixelType_Gvsp_BayerGR8 = 0  # type: ignore
    PixelType_Gvsp_YUV422_Packed = 0  # type: ignore
    PixelType_Gvsp_YUV422_YUYV_Packed = 0  # type: ignore
    MV_OK = 0  # type: ignore
    HIK_SDK_AVAILABLE = False
    HIK_SDK_IMPORT_ERROR: Optional[Exception] = ImportError("HIK SDK 模块未找到")
else:
    from MvCameraControl_class import MvCamera
    from CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT_INFO_EX,
        MVCC_INTVALUE,
        MV_CC_PIXEL_CONVERT_PARAM,
    )
    from CameraParams_const import MV_GIGE_DEVICE, MV_ACCESS_Exclusive
    from PixelType_header import (
        PixelType_Gvsp_BGR8_Packed,
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )
    from MvErrorDefine_const import MV_OK

    HIK_SDK_AVAILABLE = True
    HIK_SDK_IMPORT_ERROR = None


_hik_sdk_lock = threading.Lock()
_hik_sdk_refcount = 0


def _hik_sdk_acquire() -> bool:
    global _hik_sdk_refcount
    if not HIK_SDK_AVAILABLE or MvCamera is None:
        return False
    with _hik_sdk_lock:
        if _hik_sdk_refcount == 0:
            try:
                ret = MvCamera.MV_CC_Initialize()
            except Exception as exc:
                print(f"[HIK] SDK 初始化异常: {exc}")
                return False
            if ret != MV_OK:
                print(f"[HIK] SDK 初始化失败: 0x{ret:08X}")
                return False
        _hik_sdk_refcount += 1
        return True


def _hik_sdk_release():
    global _hik_sdk_refcount
    if not HIK_SDK_AVAILABLE or MvCamera is None:
        return
    with _hik_sdk_lock:
        if _hik_sdk_refcount <= 0:
            _hik_sdk_refcount = 0
            return
        _hik_sdk_refcount -= 1
        if _hik_sdk_refcount == 0:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass


class _HikSDKGuard:
    def __enter__(self):
        self._acquired = _hik_sdk_acquire()
        return self._acquired

    def __exit__(self, exc_type, exc, tb):
        if self._acquired:
            _hik_sdk_release()
