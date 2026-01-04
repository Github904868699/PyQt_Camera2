# -*- coding: utf-8 -*-

import os
import sys
import struct
import ctypes
import socketserver
import threading
from ctypes import POINTER, byref, cast, c_ubyte
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

import time
import socket

# Windows DLL/OpenMP 兼容性兜底（避免依赖初始化失败而导入失败）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import cv2

from config.loader import CONFIG_PATH, load_config, resource_path
from inference.onnx_yolo import ORT_AVAILABLE, ORT_IMPORT_ERROR, OnnxYoloModel, list_onnx_models
from sdk.hik_camera import (
    HIK_SDK_AVAILABLE,
    HIK_SDK_IMPORT_ERROR,
    MV_ACCESS_Exclusive,
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_PIXEL_CONVERT_PARAM,
    MV_FRAME_OUT_INFO_EX,
    MV_GIGE_DEVICE,
    MV_OK,
    MVCC_INTVALUE,
    MvCamera,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_RGB8_Packed,
    PixelType_Gvsp_YUV422_Packed,
    PixelType_Gvsp_YUV422_YUYV_Packed,
    _HikSDKGuard,
)
from ui.callbacks import UiCallbacks
from ui.qt_main_window import setup_main_window_ui

TARGET_DISPLAY_WIDTH = 1280
UI_TARGET_FPS = 15.0
UI_PAINT_FPS = 12.0
TRIGGER_REGISTER_ADDR_1 = 0
TRIGGER_REGISTER_ADDR_2 = 1
RESULT_REGISTER_ADDR_1 = 2
RESULT_REGISTER_ADDR_2 = 3
PULSE_REGISTER_ADDR_1 = 4
PULSE_REGISTER_ADDR_2 = 5

APP_TITLE = "Camera"
APP_ICON = "Camera.ico"

# ---------------------- Helpers ----------------------


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def _usb_backend():
    if os.name == "nt":
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    return cv2.CAP_ANY


def scan_usb_indices(max_index: int = 10) -> List[int]:
    backend = _usb_backend()
    found: List[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        ok = bool(cap and cap.isOpened())
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if ok:
            found.append(i)
    return found or [0]


def _hik_decode_text(buf) -> str:
    try:
        raw = bytes(bytearray(buf))
    except Exception:
        return ""
    raw = raw.split(b"\0", 1)[0]
    try:
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _hik_uint_to_ip(value: int) -> str:
    try:
        return socket.inet_ntoa(struct.pack(">I", value))
    except Exception:
        return ""


def _hik_is_same_lan(info, local_ip_int: Optional[int]) -> bool:
    if not local_ip_int:
        return False
    try:
        gige = info.SpecialInfo.stGigEInfo
        cam_ip = int(gige.nCurrentIp)
        mask = int(gige.nCurrentSubNetMask) or 0xFFFFFFFF
        return (local_ip_int & mask) == (cam_ip & mask)
    except Exception:
        return False


def _hik_format_device_name(info, local_ip_int: Optional[int]) -> str:
    try:
        gige = info.SpecialInfo.stGigEInfo
        name = _hik_decode_text(gige.chUserDefinedName) or _hik_decode_text(gige.chModelName)
        ip = _hik_uint_to_ip(int(gige.nCurrentIp))
        if name and ip:
            base = f"{name} ({ip})"
        elif ip:
            base = f"Hik GIGE ({ip})"
        else:
            base = name or "Hik GIGE"
        if _hik_is_same_lan(info, local_ip_int):
            return base + " | 本地网段"
        return base
    except Exception:
        return "Hik GIGE"


def enumerate_hik_devices() -> List[tuple]:
    """Return available HIK devices sorted with same-LAN ones first."""

    if not HIK_SDK_AVAILABLE or MvCamera is None:
        return []

    dev_list = MV_CC_DEVICE_INFO_LIST()

    with _HikSDKGuard() as ok:
        if not ok:
            return []

        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, dev_list)
        if ret != MV_OK or dev_list.nDeviceNum == 0:
            return []

    local_ip_int = None
    try:
        local_ip_int = struct.unpack(">I", socket.inet_aton(get_local_ip()))[0]
    except Exception:
        local_ip_int = None

    candidates = []
    for idx in range(int(dev_list.nDeviceNum)):
        ptr = dev_list.pDeviceInfo[idx]
        if not ptr:
            continue
        info_copy = MV_CC_DEVICE_INFO()
        ctypes.memmove(byref(info_copy), byref(ptr.contents), ctypes.sizeof(MV_CC_DEVICE_INFO))
        if not MvCamera.MV_CC_IsDeviceAccessible(info_copy, MV_ACCESS_Exclusive):
            continue
        display = _hik_format_device_name(info_copy, local_ip_int)
        same_lan = _hik_is_same_lan(info_copy, local_ip_int)
        candidates.append((info_copy, display, same_lan))

    candidates.sort(key=lambda item: (not item[2], item[1]))
    return candidates


# ---------------------- Modbus ----------------------
class ModbusRegisterModel:
    def __init__(self, size: int = 16):
        self._lock = threading.Lock()
        self._regs = [0] * max(size, 2)

    def read(self, addr: int, count: int) -> List[int]:
        with self._lock:
            if addr < 0:
                return [0] * max(count, 0)
            end = addr + count
            slice_regs = self._regs[addr:end]
            if len(slice_regs) < count:
                slice_regs.extend([0] * (count - len(slice_regs)))
            return list(slice_regs)

    def write(self, addr: int, values: List[int]):
        if addr < 0:
            return
        with self._lock:
            end = addr + len(values)
            if end > len(self._regs):
                self._regs.extend([0] * (end - len(self._regs)))
            for i, v in enumerate(values):
                self._regs[addr + i] = v & 0xFFFF

    def set_register(self, addr: int, value: int):
        self.write(addr, [value])


class ModbusRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        client_ip = self.client_address[0] if self.client_address else ""
        if client_ip:
            self.server.set_client(client_ip)
        try:
            while True:
                header = self._recvn(7)
                if not header:
                    break
                try:
                    tid, pid, length = struct.unpack(">HHH", header[:6])
                except struct.error:
                    break
                unit = header[6]
                if length <= 0:
                    continue
                payload = self._recvn(length - 1)
                if payload is None:
                    break
                if not payload:
                    continue
                function = payload[0]
                data = payload[1:]
                response_pdu = self._handle_function(function, data)
                if response_pdu is None:
                    continue
                mbap = struct.pack(">HHHB", tid, 0, len(response_pdu) + 1, unit)
                try:
                    self.request.sendall(mbap + response_pdu)
                except Exception:
                    break
        finally:
            if client_ip:
                self.server.clear_client(client_ip)

    def _recvn(self, size: int):
        buf = b""
        while len(buf) < size:
            chunk = self.request.recv(size - len(buf))
            if not chunk:
                return None if not buf else buf
            buf += chunk
        return buf

    def _handle_function(self, function: int, data: bytes) -> bytes | None:
        try:
            if function == 3:  # Read Holding Registers
                if len(data) < 4:
                    raise ValueError
                addr, count = struct.unpack(">HH", data[:4])
                regs = self.server.model.read(addr, count)
                payload = struct.pack(">B", len(regs) * 2)
                if regs:
                    payload += struct.pack(">" + "H" * len(regs), *regs)
                return bytes([function]) + payload
            elif function == 6:  # Write Single Register
                if len(data) < 4:
                    raise ValueError
                addr, value = struct.unpack(">HH", data[:4])
                self.server.model.write(addr, [value])
                if self.server.on_write:
                    self.server.on_write(addr, value & 0xFFFF)
                return bytes([function]) + data[:4]
            elif function == 16:  # Write Multiple Registers
                if len(data) < 5:
                    raise ValueError
                addr, count, byte_count = struct.unpack(">HHB", data[:5])
                expected = count * 2
                if byte_count != expected or len(data[5:]) < expected:
                    raise ValueError
                raw = data[5:5 + expected]
                values = list(struct.unpack(">" + "H" * count, raw))
                self.server.model.write(addr, values)
                if self.server.on_write:
                    for i, v in enumerate(values):
                        self.server.on_write(addr + i, v & 0xFFFF)
                return bytes([function]) + struct.pack(">HH", addr, count)
            else:
                return bytes([function | 0x80, 1])
        except Exception:
            return bytes([function | 0x80, 3])


class ModbusTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, host: str, port: int, model: ModbusRegisterModel, on_write=None):
        self.model = model
        self.on_write = on_write
        self._client_lock = threading.Lock()
        self._client_addr: Optional[str] = None
        super().__init__((host, port), ModbusRequestHandler)

    @property
    def client_addr(self) -> Optional[str]:
        with self._client_lock:
            return self._client_addr

    def set_client(self, addr: str):
        with self._client_lock:
            self._client_addr = addr

    def clear_client(self, addr: str):
        with self._client_lock:
            if self._client_addr == addr:
                self._client_addr = None


def start_modbus_server(host: str, port: int, model: ModbusRegisterModel, on_write=None):
    server = ModbusTCPServer(host, port, model, on_write=on_write)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[MODBUS] listen on {host}:{port}")
    return server


# ---------------------- Grabbers ----------------------
class HikGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal = QtCore.pyqtSignal(str)
    errorSignal = QtCore.pyqtSignal(str)
    autoAdjustSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None, device_index: int = 0):
        super().__init__(parent)
        if not HIK_SDK_AVAILABLE or MvCamera is None:
            reason = str(HIK_SDK_IMPORT_ERROR) if HIK_SDK_IMPORT_ERROR else "未检测到海康 SDK"
            raise RuntimeError(f"海康 SDK 未就绪: {reason}")

        self.device_index = int(device_index)

        self.camera: Optional["MvCamera"] = None
        self._running = False

        self._data_buf = None
        self._data_ptr = None
        self._convert_buf = None
        self._convert_ptr = None
        self._convert_buf_size = 0
        self._payload_size = 0

        self._last_emit_ts = 0.0
        self._local_ip_int = self._ip_to_uint(get_local_ip())

        self._bayer_types = {
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerGR8,
        }
        self._last_stream_error = 0
        self._last_convert_error = 0
        self._last_unsupported_pixel = 0

        self.autoAdjustSignal.connect(self._handle_auto_adjust)

    @staticmethod
    def _ip_to_uint(ip: str) -> Optional[int]:
        try:
            return struct.unpack(">I", socket.inet_aton(ip))[0]
        except Exception:
            return None

    @staticmethod
    def _uint_to_ip(value: int) -> str:
        try:
            return socket.inet_ntoa(struct.pack(">I", value))
        except Exception:
            return ""

    @staticmethod
    def _decode_text(buf) -> str:
        try:
            raw = bytes(bytearray(buf))
        except Exception:
            return ""
        raw = raw.split(b"\0", 1)[0]
        try:
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _is_same_lan(self, info) -> bool:
        if not self._local_ip_int:
            return False
        try:
            gige = info.SpecialInfo.stGigEInfo
            cam_ip = int(gige.nCurrentIp)
            mask = int(gige.nCurrentSubNetMask) or 0xFFFFFFFF
            return (self._local_ip_int & mask) == (cam_ip & mask)
        except Exception:
            return False

    def _format_device_name(self, info) -> str:
        try:
            gige = info.SpecialInfo.stGigEInfo
            name = self._decode_text(gige.chUserDefinedName) or self._decode_text(gige.chModelName)
            ip = self._uint_to_ip(int(gige.nCurrentIp))
            if name and ip:
                return f"{name} ({ip})"
            if ip:
                return f"Hik GIGE ({ip})"
            return name or "Hik GIGE"
        except Exception:
            return "Hik GIGE"

    def _prepare_payload(self):
        payload = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("PayloadSize", payload)
        if ret != MV_OK or int(payload.nCurValue) <= 0:
            raise RuntimeError(f"获取 PayloadSize 失败: 0x{ret:08X}")
        self._payload_size = int(payload.nCurValue)
        self._data_buf = (c_ubyte * self._payload_size)()
        self._data_ptr = cast(self._data_buf, POINTER(c_ubyte))

    def _get_int_value(self, key: str) -> Optional[int]:
        if not self.camera:
            return None
        value = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue(key, value)
        if ret == MV_OK:
            return int(value.nCurValue)
        return None

    def _ensure_convert_buffer(self, size: int):
        if self._convert_buf_size < size:
            self._convert_buf = (c_ubyte * size)()
            self._convert_ptr = cast(self._convert_buf, POINTER(c_ubyte))
            self._convert_buf_size = size

    def _convert_frame(self, frame_info: "MV_FRAME_OUT_INFO_EX") -> Optional[np.ndarray]:
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        frame_len = int(frame_info.nFrameLen)
        pixel_type = int(frame_info.enPixelType)

        if frame_len <= 0 or width <= 0 or height <= 0:
            return None

        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            return arr.reshape(height, width, 3).copy()

        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            rgb = arr.reshape(height, width, 3)
            return rgb[:, :, ::-1].copy()

        if pixel_type == PixelType_Gvsp_Mono8:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            gray = arr.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if pixel_type in self._bayer_types or pixel_type in {
            PixelType_Gvsp_YUV422_Packed,
            PixelType_Gvsp_YUV422_YUYV_Packed,
        }:
            dst_size = width * height * 3
            self._ensure_convert_buffer(dst_size)

            convert_param = MV_CC_PIXEL_CONVERT_PARAM()
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.enSrcPixelType = pixel_type
            convert_param.pSrcData = self._data_ptr
            convert_param.nSrcDataLen = frame_len
            convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            convert_param.pDstBuffer = self._convert_ptr
            convert_param.nDstBufferSize = dst_size
            convert_param.nDstLen = dst_size

            ret = self.camera.MV_CC_ConvertPixelType(convert_param)
            if ret != MV_OK:
                if self._last_convert_error != ret:
                    self.infoSignal.emit(f"[HIK] 像素转换失败: 0x{ret:08X}")
                    self._last_convert_error = ret
                return None

            self._last_convert_error = 0
            arr = np.frombuffer(self._convert_buf, dtype=np.uint8, count=dst_size)
            return arr.reshape(height, width, 3).copy()

        if self._last_unsupported_pixel != pixel_type:
            self.infoSignal.emit(f"[HIK] 不支持的像素格式: 0x{pixel_type:08X}")
            self._last_unsupported_pixel = pixel_type
        return None

    @QtCore.pyqtSlot()
    def _handle_auto_adjust(self):
        cam = getattr(self, "camera", None)
        if not cam:
            self.infoSignal.emit("[HIK] 相机未就绪，无法自动调节")
            return

        def _set_enum(name: str, value: int):
            try:
                ret = cam.MV_CC_SetEnumValue(name, value)
                if ret != MV_OK:
                    self.infoSignal.emit(f"[HIK] {name} 自动设置失败: 0x{ret:08X}")
                    return False
                return True
            except Exception as exc:
                self.infoSignal.emit(f"[HIK] {name} 自动设置异常: {exc}")
                return False

        ok_exp = _set_enum("ExposureAuto", 1)  # 1=Once
        ok_gain = _set_enum("GainAuto", 1)     # 1=Once
        ok_wb = _set_enum("BalanceWhiteAuto", 1)  # 1=Once

        if ok_exp or ok_gain or ok_wb:
            self.infoSignal.emit("[HIK] 已执行自动调节")
        else:
            self.infoSignal.emit("[HIK] 自动调节失败")

    def _cleanup_camera(self):
        if self.camera:
            try:
                self.camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.camera.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.camera = None

        self._data_buf = None
        self._data_ptr = None
        self._convert_buf = None
        self._convert_ptr = None
        self._convert_buf_size = 0

    def stop(self):
        self._running = False

    def run(self):
        try:
            with _HikSDKGuard() as ok:
                if not ok:
                    raise RuntimeError("初始化海康 SDK 失败")

                try:
                    devices = enumerate_hik_devices()
                    if not devices:
                        raise RuntimeError("未发现可用的海康相机")

                    if self.device_index >= len(devices):
                        raise RuntimeError(f"海康相机索引 {self.device_index + 1} 不存在")

                    device_info, display_name, _ = devices[self.device_index]
                    self.camera = MvCamera()

                    ret = self.camera.MV_CC_CreateHandle(device_info)
                    if ret != MV_OK:
                        raise RuntimeError(f"创建相机句柄失败: 0x{ret:08X}")

                    ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                    if ret != MV_OK:
                        raise RuntimeError(f"打开相机失败: 0x{ret:08X}")

                    self._prepare_payload()

                    ret = self.camera.MV_CC_StartGrabbing()
                    if ret != MV_OK:
                        raise RuntimeError(f"启动取流失败: 0x{ret:08X}")

                    width = self._get_int_value("Width")
                    height = self._get_int_value("Height")
                    if width and height:
                        self.infoSignal.emit(f"[INFO] HIK | {display_name} | {width}x{height}")
                    else:
                        self.infoSignal.emit(f"[INFO] HIK | {display_name}")

                    self._running = True
                    frame_info = MV_FRAME_OUT_INFO_EX()

                    grabbed = 0
                    last_fps_ts = time.time()
                    self._last_emit_ts = 0.0

                    while self._running:
                        ret = self.camera.MV_CC_GetOneFrameTimeout(
                            self._data_ptr, self._payload_size, frame_info, 1000
                        )
                        if ret != MV_OK:
                            if ret != self._last_stream_error:
                                self.infoSignal.emit(f"[HIK] 取流异常: 0x{ret:08X}")
                                self._last_stream_error = ret
                            continue

                        self._last_stream_error = 0
                        now = time.time()
                        grabbed += 1

                        if (now - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                            if (now - last_fps_ts) >= 1.0:
                                self.infoSignal.emit(f"[FPS] {grabbed / (now - last_fps_ts):.1f}")
                                grabbed = 0
                                last_fps_ts = now
                            continue

                        frame = self._convert_frame(frame_info)
                        if frame is None:
                            continue

                        self._last_emit_ts = now
                        self.frameSignal.emit(frame)

                        if (now - last_fps_ts) >= 1.0:
                            self.infoSignal.emit(f"[FPS] {grabbed / (now - last_fps_ts):.1f}")
                            grabbed = 0
                            last_fps_ts = now

                        QtCore.QThread.msleep(1)
                finally:
                    self._cleanup_camera()

        except Exception as exc:
            self._running = False
            self.infoSignal.emit(f"[HIK] {exc}")
            self.errorSignal.emit(str(exc))


class UsbGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal = QtCore.pyqtSignal(str)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, index: int = 0):
        super().__init__(parent)
        self.index = int(index)
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._last_emit_ts = 0.0

    def run(self):
        try:
            backend = _usb_backend()
            self.cap = cv2.VideoCapture(self.index, backend)
            if (not self.cap or not self.cap.isOpened()) and backend != cv2.CAP_ANY:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = cv2.VideoCapture(self.index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"无法打开 USB 摄像头 (index={self.index})")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width > 0 and height > 0:
                self.infoSignal.emit(f"[INFO] USB | index={self.index} | {width}x{height}")
            else:
                self.infoSignal.emit(f"[INFO] USB | index={self.index}")

            self._running = True
            t0 = time.time()
            grabbed = 0

            while self._running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    QtCore.QThread.msleep(5)
                    continue

                grabbed += 1

                if frame.shape[1] > TARGET_DISPLAY_WIDTH:
                    scale = TARGET_DISPLAY_WIDTH / float(frame.shape[1])
                    frame = cv2.resize(
                        frame,
                        (TARGET_DISPLAY_WIDTH, int(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                now_ts = time.time()
                if (now_ts - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                    QtCore.QThread.msleep(5)
                    continue

                self._last_emit_ts = now_ts
                self.frameSignal.emit(frame.copy())

                now = time.time()
                if now - t0 >= 1.0:
                    self.infoSignal.emit(f"[FPS] {grabbed / (now - t0):.1f}")
                    t0 = now
                    grabbed = 0

                QtCore.QThread.msleep(1)

        except Exception as e:
            self.infoSignal.emit(f"[USB] {e}")
            self.errorSignal.emit(str(e))
        finally:
            self._stop_and_close()

    def stop(self):
        self._running = False

    def _stop_and_close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None


# ---------------------- Main UI ----------------------
class MainWindow(QtWidgets.QMainWindow):
    modbus_trigger_sig = QtCore.pyqtSignal(int)
    yolo_result_sig = QtCore.pyqtSignal(int, int, int)
    toast_sig = QtCore.pyqtSignal(str, int)

    def __init__(self):
        super().__init__()

        self.config = load_config(CONFIG_PATH)
        server_cfg = self.config.get("server", {})
        self.class_map: Dict[str, int] = {k: int(v) for k, v in self.config.get("class_map", {}).items()}
        self.models: List[str] = list_onnx_models()
        self.default_model: Optional[str] = self.models[0] if self.models else None
        self.modbus_host = str(server_cfg.get("host", "0.0.0.0") or "0.0.0.0")
        self.modbus_port = int(server_cfg.get("port", 502))
        self.modbus_model = ModbusRegisterModel(size=8)
        self.modbus_server = None
        self.modbus_error: Optional[str] = None
        self.modbus_trigger_sig.connect(self._on_modbus_trigger)
        self.yolo_result_sig.connect(self._on_yolo_result)
        self.toast_sig.connect(self._show_toast)
        self._last_trigger_values = {
            TRIGGER_REGISTER_ADDR_1: 0,
            TRIGGER_REGISTER_ADDR_2: 0,
        }
        self._trigger_active = {1: False, 2: False}
        self._recognition_in_progress = {1: False, 2: False}
        self._executor = ThreadPoolExecutor(max_workers=1)

        # 默认摄像头配置（双路独立）
        self.slot_sources: Dict[int, str] = {1: "hik", 2: "hik"}
        self.hik_indices: Dict[int, int] = {1: 0, 2: 1}
        self.usb_indices_selected: Dict[int, int] = {1: 0, 2: 1}
        self.usb_indices: List[int] = []
        self.hik_devices: List[tuple] = []
        self.active_slots: List[int] = []

        # 默认 YOLO 配置（双路独立启用与模型）
        self.yolo_conf = 0.5
        self.slot_yolo_enabled: Dict[int, bool] = {1: False, 2: False}
        self.slot_model_paths: Dict[int, Optional[str]] = {1: self.default_model, 2: self.default_model}
        self.yolo_model_cache: Dict[str, OnnxYoloModel] = {}
        self.yolo_style = 0     # 0~7 不同样式

        self.last_frame_bgrs: Dict[int, np.ndarray] = {}
        self._last_paint_ts: Dict[int, float] = {}
        self._last_frame_ts: Dict[int, float] = {1: 0.0, 2: 0.0}
        self._last_restart_ts: Dict[int, float] = {1: 0.0, 2: 0.0}
        self.grabbers: Dict[int, QtCore.QThread] = {}

        self._start_modbus_server(self.modbus_host)

        callbacks = UiCallbacks(
            on_slot_source_changed=self._on_slot_source_changed,
            on_refresh_hik=self._refresh_hik_devices,
            on_refresh_usb=self._refresh_usb_indices,
            on_hik_index_changed=self._on_hik_index_changed,
            on_usb_index_changed=self._on_usb_index_changed,
            on_reopen_camera=self.reopen_camera,
            on_stop_camera=self.stop_camera,
            on_auto_adjust=self._on_auto_adjust_clicked,
            on_yolo_enabled_changed=self._on_yolo_enabled_changed,
            on_yolo_model_changed=self._on_yolo_model_changed,
            on_yolo_conf_changed=self._on_yolo_conf_changed,
            on_yolo_style_changed=self._on_yolo_style_changed,
        )
        setup_main_window_ui(self, callbacks, title=APP_TITLE)

        self._refresh_hik_devices()
        self._refresh_usb_indices()
        self._update_all_slot_controls()
        self._start_camera()

    # ---------- UI 逻辑 ----------
    def _video_label_for(self, cam_index: int) -> Optional[QtWidgets.QLabel]:
        return self.video_lbl1 if cam_index == 1 else self.video_lbl2 if cam_index == 2 else None

    def _slot_source(self, slot: int) -> str:
        widget = self.slot_widgets.get(slot, {}).get("source")
        if widget is None:
            return "hik"
        return str(widget.currentData() or "hik")

    def _update_slot_stack(self, slot: int):
        widgets = self.slot_widgets.get(slot, {})
        idx_stack: QtWidgets.QStackedWidget = widgets.get("idx_stack")  # type: ignore
        if idx_stack is None:
            return
        is_usb = self._slot_source(slot) == "usb"
        target = idx_stack.widget(1 if is_usb else 0)
        if target:
            idx_stack.setCurrentWidget(target)

    def _update_slot_controls(self, slot: int):
        widgets = self.slot_widgets.get(slot, {})
        is_hik = self._slot_source(slot) == "hik"
        self._update_slot_stack(slot)

        hik_combo: QtWidgets.QComboBox = widgets.get("hik_combo")  # type: ignore
        usb_combo: QtWidgets.QComboBox = widgets.get("usb_combo")  # type: ignore
        btn_hik_refresh: QtWidgets.QPushButton = widgets.get("hik_refresh")  # type: ignore
        btn_usb_refresh: QtWidgets.QPushButton = widgets.get("usb_refresh")  # type: ignore
        btn_auto: QtWidgets.QPushButton = widgets.get("btn_auto")  # type: ignore

        if hik_combo:
            hik_combo.setEnabled(is_hik and len(self.hik_devices) > 1)
        if usb_combo:
            usb_combo.setEnabled((not is_hik) and len(self.usb_indices) > 1)
        if btn_hik_refresh:
            btn_hik_refresh.setEnabled(True)
        if btn_usb_refresh:
            btn_usb_refresh.setEnabled(True)
        if btn_auto:
            btn_auto.setEnabled(is_hik)

    def _update_all_slot_controls(self):
        for slot in (1, 2):
            self._update_slot_controls(slot)

    def _update_video_visibility(self):
        for slot in (1, 2):
            lbl = self._video_label_for(slot)
            if slot not in self.active_slots and lbl:
                lbl.clear()
                lbl.setText("无信号")

    def _refresh_hik_devices(self):
        try:
            devices = enumerate_hik_devices()
        except Exception as exc:
            devices = []
            self._toast(f"海康枚举失败: {exc}")
        self.hik_devices = devices

        for slot in (1, 2):
            combo: QtWidgets.QComboBox = self.slot_widgets.get(slot, {}).get("hik_combo")  # type: ignore
            if combo is None:
                continue
            blocker = QtCore.QSignalBlocker(combo)
            combo.clear()
            for idx, _info in enumerate(self.hik_devices):
                combo.addItem(f"摄像头{idx + 1}", idx)
            if self.hik_devices and combo.count() > 0:
                desired = self.hik_indices.get(slot, 0)
                idx = combo.findData(int(desired))
                if idx < 0:
                    idx = 0
                combo.setCurrentIndex(idx)
                self.hik_indices[slot] = int(combo.itemData(idx) or 0)
            del blocker
            try:
                combo.currentIndexChanged.disconnect()
            except Exception:
                pass
            combo.currentIndexChanged.connect(lambda i, s=slot: self._on_hik_index_changed(s, i))

        self._update_all_slot_controls()

    def _refresh_usb_indices(self):
        indices = scan_usb_indices(max_index=10)
        self.usb_indices = list(indices)
        for slot in (1, 2):
            combo: QtWidgets.QComboBox = self.slot_widgets.get(slot, {}).get("usb_combo")  # type: ignore
            if combo is None:
                continue
            blocker = QtCore.QSignalBlocker(combo)
            combo.clear()
            for i in indices:
                combo.addItem(str(i), int(i))
            if combo.count() > 0:
                desired = self.usb_indices_selected.get(slot, 0)
                idx = combo.findData(int(desired))
                if idx < 0:
                    idx = 0
                combo.setCurrentIndex(idx)
                self.usb_indices_selected[slot] = int(combo.itemData(idx) or 0)
            del blocker
            try:
                combo.currentIndexChanged.disconnect()
            except Exception:
                pass
            combo.currentIndexChanged.connect(lambda i, s=slot: self._on_usb_index_changed(s, i))

        self._update_all_slot_controls()

    def _on_slot_source_changed(self, slot: int, _index: int):
        self.slot_sources[slot] = self._slot_source(slot)
        self._update_slot_controls(slot)
        self._start_slot(slot)

    def _on_usb_index_changed(self, slot: int, index: int):
        if index < 0:
            return
        combo: QtWidgets.QComboBox = self.slot_widgets.get(slot, {}).get("usb_combo")  # type: ignore
        if combo is None:
            return
        data = combo.itemData(index)
        if data is None:
            return
        self.usb_indices_selected[slot] = int(data)
        if self._slot_source(slot) == "usb":
            self._start_slot(slot)

    def _on_hik_index_changed(self, slot: int, index: int):
        if index < 0:
            return
        combo: QtWidgets.QComboBox = self.slot_widgets.get(slot, {}).get("hik_combo")  # type: ignore
        if combo is None:
            return
        data = combo.itemData(index)
        if data is None:
            return
        self.hik_indices[slot] = int(data)
        if self._slot_source(slot) == "hik":
            self._start_slot(slot)

    def _on_auto_adjust_clicked(self, slot: int):
        target = None
        grabber = self.grabbers.get(slot)
        if grabber and isinstance(grabber, HikGrabber):
            target = grabber
        if target is None:
            self._toast(f"摄像头{slot} 当前非海康相机，无法自动调节")
            return
        target.autoAdjustSignal.emit()
        self._toast(f"摄像头{slot} 正在自动调节...")

    # ---------- YOLO 控制 ----------
    def _on_yolo_enabled_changed(self, slot: int, state: int):
        enabled = (state == QtCore.Qt.Checked)
        self.slot_yolo_enabled[slot] = enabled
        if enabled:
            self._ensure_yolo_model(self.slot_model_paths.get(slot))

    def _on_yolo_conf_changed(self, value: float):
        self.yolo_conf = max(0.0, min(1.0, float(value)))

    def _on_yolo_model_changed(self, cam_index: int, index: int):
        if index < 0:
            return

        combo: QtWidgets.QComboBox = self.slot_widgets.get(cam_index, {}).get("combo_model")  # type: ignore
        if combo is None:
            return
        data = combo.itemData(index)
        model_path = str(data or combo.currentText() or "").strip()
        if not model_path:
            return

        if self.slot_model_paths.get(cam_index) != model_path:
            self.slot_model_paths[cam_index] = model_path
            # 清除旧缓存，按需重新加载
            norm_path = self._normalized_model_path(model_path)
            if norm_path in self.yolo_model_cache:
                self.yolo_model_cache.pop(norm_path, None)
            if self.slot_yolo_enabled.get(cam_index):
                self._ensure_yolo_model(model_path)

    def _on_yolo_style_changed(self, index: int):
        data = self.combo_style.itemData(index)
        self.yolo_style = int(data) if data is not None else int(index)

    def _normalized_model_path(self, model_path: str) -> str:
        candidate = model_path
        if not os.path.isabs(candidate):
            res = resource_path(candidate)
            if os.path.exists(res):
                candidate = res
        return candidate

    def _ensure_yolo_model(self, model_path: Optional[str] = None) -> bool:
        if not ORT_AVAILABLE:
            self._toast(f"[YOLO] 导入 onnxruntime 失败: {ORT_IMPORT_ERROR}")
            return False

        selected = model_path or self.default_model
        if not selected:
            self._toast("[YOLO] 未选择模型文件")
            return False

        path = self._normalized_model_path(selected)
        if not path.lower().endswith(".onnx"):
            self._toast("[YOLO] 请使用 .onnx 模型文件")
            return False

        if path in self.yolo_model_cache:
            return True

        if not os.path.exists(path):
            self._toast(f"[YOLO] 模型文件不存在: {path}")
            return False

        try:
            self.yolo_model_cache[path] = OnnxYoloModel(path)
            self._toast(f"[YOLO] 模型已加载: {os.path.basename(path)}", ms=2000)
            return True
        except Exception as exc:
            self._toast(f"[YOLO] 加载模型失败: {exc}")
            return False

    def _model_for_cam(self, cam_index: int) -> Optional[OnnxYoloModel]:
        model_path = self.slot_model_paths.get(cam_index) or self.default_model
        if not model_path:
            self._toast("[YOLO] 未选择模型文件")
            return None

        norm_path = self._normalized_model_path(model_path)
        if norm_path not in self.yolo_model_cache:
            if not self._ensure_yolo_model(norm_path):
                return None
        return self.yolo_model_cache.get(norm_path)

    def _apply_yolo(self, cam_index: int, frame_bgr: np.ndarray) -> np.ndarray:
        """在 BGR 图像上做 YOLO 推理并画框，只显示高于当前阈值的目标。
        同时根据分辨率自动调整字体大小和线宽。
        """
        model = self._model_for_cam(cam_index)
        if model is None:
            return frame_bgr

        try:
            detections = model.predict(frame_bgr, conf_thres=float(self.yolo_conf))
            if not detections:
                return frame_bgr

            overlay = frame_bgr.copy()
            style = int(getattr(self, "yolo_style", 0))
            font = cv2.FONT_HERSHEY_SIMPLEX

            # ---------- 根据图像高度自适应字体 & 线宽 ----------
            h_img, w_img = overlay.shape[:2]
            base_ref_h = 480.0  # 480 高度基准
            scale_factor = h_img / base_ref_h
            scale_factor = max(0.7, min(5.0, scale_factor))

            base_font_scale = 0.6
            font_scale = base_font_scale * scale_factor
            base_thickness = max(1, int(round(1 * scale_factor)))

            line_thick_1 = max(1, int(round(1 * scale_factor)))
            line_thick_2 = max(1, int(round(2 * scale_factor)))
            line_thick_4 = max(2, int(round(4 * scale_factor)))

            # ----- 颜色方案 -----
            palette_yolo = [
                (0, 255, 255),
                (0, 0, 255),
                (255, 0, 0),
                (0, 255, 0),
                (255, 0, 255),
                (0, 128, 255),
                (255, 255, 0),
                (128, 0, 255),
                (255, 0, 128),
                (0, 255, 128),
            ]
            palette_a = [
                (80, 180, 200),
                (80, 160, 120),
                (180, 160, 100),
                (120, 150, 210),
                (150, 120, 180),
                (120, 120, 120),
                (70, 140, 170),
                (140, 140, 200),
                (160, 140, 120),
                (100, 170, 140),
            ]
            palette_b = [(b // 2 + 20, g // 2 + 20, r // 2 + 20) for (b, g, r) in palette_a]
            palette_neon = [
                (0, 255, 191),
                (255, 0, 191),
                (0, 191, 255),
                (191, 255, 0),
                (255, 128, 0),
                (191, 0, 255),
                (0, 255, 128),
                (255, 0, 128),
                (128, 255, 0),
                (0, 128, 255),
            ]

            def color_for_class(cls_id: int, palette: List[tuple]) -> tuple:
                if not palette:
                    return (0, 255, 0)
                if cls_id < 0:
                    return palette[0]
                return palette[int(cls_id) % len(palette)]

            def pick_text_color(bg_bgr: tuple) -> tuple:
                """根据背景亮度自动选黑/白字，防止看不清。"""
                b, g, r = bg_bgr
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                return (0, 0, 0) if lum > 160 else (255, 255, 255)

            for det in detections:
                score = float(det["score"])
                cls_id = int(det["cls_id"])
                label = model.label_for(cls_id)

                x1, y1, x2, y2 = [int(v) for v in det["bbox"].tolist()]
                text = f"{label} {score:.2f}"

                x1 = max(0, min(w_img - 1, x1))
                x2 = max(0, min(w_img - 1, x2))
                y1 = max(0, min(h_img - 1, y1))
                y2 = max(0, min(h_img - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                box_color = (0, 255, 0)
                label_bg_color = (0, 0, 0)
                txt_color = (255, 255, 255)
                line_thick = line_thick_2

                # ---- 不同样式画框 ----
                if style == 0:
                    # 经典绿色框
                    box_color = (0, 255, 0)
                    label_bg_color = (0, 0, 0)
                    txt_color = (0, 255, 0)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 1:
                    # 相机准星 + 红点，用 palette_b（暗一点）
                    box_color = color_for_class(cls_id, palette_b)
                    label_bg_color = (0, 0, 0)
                    txt_color = (255, 255, 255)
                    line_thick = line_thick_2

                    w_box = x2 - x1
                    h_box = y2 - y1
                    line_len = max(6, min(w_box, h_box) // 4)

                    # 四角线
                    cv2.line(overlay, (x1, y1), (x1 + line_len, y1), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x1, y1), (x1, y1 + line_len), box_color, line_thick, cv2.LINE_AA)

                    cv2.line(overlay, (x2, y1), (x2 - line_len, y1), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x2, y1), (x2, y1 + line_len), box_color, line_thick, cv2.LINE_AA)

                    cv2.line(overlay, (x1, y2), (x1 + line_len, y2), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x1, y2), (x1, y2 - line_len), box_color, line_thick, cv2.LINE_AA)

                    cv2.line(overlay, (x2, y2), (x2 - line_len, y2), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x2, y2), (x2, y2 - line_len), box_color, line_thick, cv2.LINE_AA)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(overlay, (cx, cy), max(2, int(3 * scale_factor)), (0, 0, 200), -1, cv2.LINE_AA)

                elif style == 2:
                    # YOLO 原版：palette_yolo
                    box_color = color_for_class(cls_id, palette_yolo)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 3:
                    # 类别配色 A：palette_a
                    box_color = color_for_class(cls_id, palette_a)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 4:
                    # 类别配色 B：palette_b（更柔和）
                    box_color = color_for_class(cls_id, palette_b)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 5:
                    # 极简白框 + 阴影
                    box_color = (255, 255, 255)
                    label_bg_color = (0, 0, 0)
                    txt_color = (255, 255, 255)
                    line_thick = line_thick_1
                    cv2.rectangle(
                        overlay,
                        (x1 + line_thick_1, y1 + line_thick_1),
                        (x2 + line_thick_1, y2 + line_thick_1),
                        (0, 0, 0),
                        thickness=line_thick_2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 6:
                    # 霓虹边框：palette_neon
                    box_color = color_for_class(cls_id, palette_neon)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick_4, cv2.LINE_AA)
                    cv2.rectangle(
                        overlay,
                        (x1 + line_thick_1, y1 + line_thick_1),
                        (x2 - line_thick_1, y2 - line_thick_1),
                        box_color,
                        line_thick_1,
                        cv2.LINE_AA,
                    )
                    line_thick = line_thick_2

                else:
                    # style == 7: 半透明填充框（用 palette_a）
                    box_color = color_for_class(cls_id, palette_a)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    roi = overlay[y1:y2, x1:x2]
                    color_layer = np.full_like(roi, box_color, dtype=np.uint8)
                    cv2.addWeighted(color_layer, 0.25, roi, 0.75, 0, roi)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), line_thick_1, cv2.LINE_AA)

                # ---- 标签文字 ----
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, base_thickness)
                text_x = x1
                text_y = y1 - 4
                if text_y - th - baseline < 0:
                    text_y = y1 + th + baseline + 4

                bg_x1 = text_x
                bg_y1 = text_y - th - baseline
                bg_x2 = text_x + tw
                bg_y2 = text_y + baseline

                bg_x1 = max(0, min(w_img - 1, bg_x1))
                bg_y1 = max(0, min(h_img - 1, bg_y1))
                bg_x2 = max(0, min(w_img - 1, bg_x2))
                bg_y2 = max(0, min(h_img - 1, bg_y2))

                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                    cv2.rectangle(
                        overlay,
                        (bg_x1, bg_y1),
                        (bg_x2, bg_y2),
                        label_bg_color,
                        thickness=-1,
                    )

                cv2.putText(
                    overlay,
                    text,
                    (bg_x1, bg_y2 - baseline),
                    font,
                    font_scale,
                    txt_color,
                    base_thickness,
                    cv2.LINE_AA,
                )

            return overlay
        except Exception as exc:
            self._toast(f"[YOLO] 推理异常: {exc}")
            return frame_bgr

    def _run_yolo_recognition(self, cam_index: int) -> int:
        frame = self.last_frame_bgrs.get(cam_index)
        if frame is None:
            self._toast(f"摄像头{cam_index} 未捕获画面")
            return 0

        model = self._model_for_cam(cam_index)
        if model is None:
            return 0xFF

        try:
            detections = model.predict(frame, conf_thres=float(self.yolo_conf))
        except Exception as exc:
            self._toast(f"[YOLO] 推理异常: {exc}")
            return 0xFF

        if not detections:
            return 0

        best_score = -1.0
        best_label: Optional[str] = None
        best_cls_id: int = -1

        for det in detections:
            score = float(det["score"])
            cls_id = int(det["cls_id"])
            label = model.label_for(cls_id)

            if score > best_score:
                best_score = score
                best_label = label
                best_cls_id = cls_id

        if best_label is None:
            return 0

        mapped = self.class_map.get(best_label)
        if mapped is None and best_cls_id >= 0:
            mapped = self.class_map.get(str(best_cls_id))

        if mapped is None:
            print(f"[MODBUS] 未找到类别映射: {best_label}")
            return 0

        return int(mapped) & 0xFFFF

    # ---------- Modbus ----------
    def _publish_modbus_result(self, addr: int, value: int):
        model = getattr(self, "modbus_model", None)
        if not model:
            return
        model.set_register(addr, int(value) & 0xFFFF)
        self._refresh_modbus_status()

    def _refresh_modbus_status(self):
        status_text = "未启动"
        if getattr(self, "modbus_error", None):
            status_text = f"错误: {self.modbus_error}"
        elif getattr(self, "modbus_server", None):
            client_addr = getattr(self.modbus_server, "client_addr", None)
            if client_addr:
                status_text = f"{client_addr}:{self.modbus_port}"
            else:
                status_text = "未连接"

        if getattr(self, "lbl_modbus_status", None) is not None:
            self.lbl_modbus_status.setText(status_text)

        reg_val_0 = None
        reg_val_1 = None
        reg_val_2 = None
        reg_val_3 = None
        model = getattr(self, "modbus_model", None)
        if model:
            try:
                vals0 = model.read(TRIGGER_REGISTER_ADDR_1, 1)
                reg_val_0 = vals0[0] if vals0 else None
                vals01 = model.read(TRIGGER_REGISTER_ADDR_2, 1)
                reg_val_1 = vals01[0] if vals01 else None
                vals2 = model.read(RESULT_REGISTER_ADDR_1, 1)
                reg_val_2 = vals2[0] if vals2 else None
                vals3 = model.read(RESULT_REGISTER_ADDR_2, 1)
                reg_val_3 = vals3[0] if vals3 else None
            except Exception:
                reg_val_0 = None
                reg_val_1 = None
                reg_val_2 = None
                reg_val_3 = None

        reg_text_0 = "—" if reg_val_0 is None else f"{reg_val_0}"
        reg_text_01 = "—" if reg_val_1 is None else f"{reg_val_1}"
        reg_text_2 = "—" if reg_val_2 is None else f"{reg_val_2}"
        reg_text_3 = "—" if reg_val_3 is None else f"{reg_val_3}"
        if getattr(self, "lbl_modbus_reg0", None) is not None:
            self.lbl_modbus_reg0.setText(reg_text_0)
        if getattr(self, "lbl_modbus_reg01", None) is not None:
            self.lbl_modbus_reg01.setText(reg_text_01)
        if getattr(self, "lbl_modbus_reg1", None) is not None:
            self.lbl_modbus_reg1.setText(reg_text_2)
        if getattr(self, "lbl_modbus_reg2", None) is not None:
            self.lbl_modbus_reg2.setText(reg_text_3)

    def _handle_recognition_request(self, cam_index: int):
        if self._recognition_in_progress.get(cam_index):
            return
        self._recognition_in_progress[cam_index] = True
        reg_addr = RESULT_REGISTER_ADDR_1 if cam_index == 1 else RESULT_REGISTER_ADDR_2
        if not self._executor:
            self._recognition_in_progress[cam_index] = False
            return

        def _task():
            return cam_index, reg_addr, self._run_yolo_recognition(cam_index)

        future = self._executor.submit(_task)
        try:
            future.cam_index = cam_index
        except Exception:
            pass
        future.add_done_callback(self._handle_yolo_future)

    def _on_modbus_write(self, addr: int, value: int):
        if addr not in (TRIGGER_REGISTER_ADDR_1, TRIGGER_REGISTER_ADDR_2):
            return
        if value == 1 and self._last_trigger_values.get(addr, 0) != 1:
            cam_index = 1 if addr == TRIGGER_REGISTER_ADDR_1 else 2
            print(f"[MODBUS] 收到触发请求 {cam_index}")
            self._last_trigger_values[addr] = 1
            self._trigger_active[cam_index] = True
            self.modbus_trigger_sig.emit(cam_index)
        elif value == 0:
            self._last_trigger_values[addr] = 0
            cam_index = 1 if addr == TRIGGER_REGISTER_ADDR_1 else 2
            self._trigger_active[cam_index] = False
            if self.modbus_model:
                pulse_addr = PULSE_REGISTER_ADDR_1 if addr == TRIGGER_REGISTER_ADDR_1 else PULSE_REGISTER_ADDR_2
                self.modbus_model.set_register(pulse_addr, 0)
                result_addr = RESULT_REGISTER_ADDR_1 if addr == TRIGGER_REGISTER_ADDR_1 else RESULT_REGISTER_ADDR_2
                self.modbus_model.set_register(result_addr, 0)

    @QtCore.pyqtSlot(int)
    def _on_modbus_trigger(self, cam_index: int):
        self._handle_recognition_request(int(cam_index))

    def _handle_yolo_future(self, future):
        try:
            cam_index, reg_addr, result_value = future.result()
        except Exception as exc:
            print(f"[YOLO] 异步推理失败: {exc}")
            try:
                cam_index = int(getattr(future, "cam_index", 0))
            except Exception:
                cam_index = 0
            if cam_index in self._recognition_in_progress:
                self._recognition_in_progress[cam_index] = False
            return
        self.yolo_result_sig.emit(int(cam_index), int(reg_addr), int(result_value))

    @QtCore.pyqtSlot(int, int, int)
    def _on_yolo_result(self, cam_index: int, reg_addr: int, result_value: int):
        if not self._trigger_active.get(cam_index, False):
            self._recognition_in_progress[cam_index] = False
            return
        self._publish_modbus_result(reg_addr, result_value)
        if self.modbus_model:
            pulse_addr = PULSE_REGISTER_ADDR_1 if cam_index == 1 else PULSE_REGISTER_ADDR_2
            self.modbus_model.set_register(pulse_addr, 1)
        self._recognition_in_progress[cam_index] = False
        if self._trigger_active.get(cam_index):
            QtCore.QTimer.singleShot(0, lambda idx=cam_index: self._handle_recognition_request(idx))

    def _stop_modbus_server(self):
        if not getattr(self, "modbus_server", None):
            return
        try:
            self.modbus_server.shutdown()
        except Exception:
            pass
        try:
            self.modbus_server.server_close()
        except Exception:
            pass
        self.modbus_server = None
        self._refresh_modbus_status()

    def _start_modbus_server(self, host: Optional[str] = None):
        if host is not None:
            self.modbus_host = host
        self._stop_modbus_server()
        self.modbus_error = None
        try:
            self.modbus_server = start_modbus_server(
                self.modbus_host, self.modbus_port, self.modbus_model, on_write=self._on_modbus_write
            )
        except Exception as exc:
            print(f"[MODBUS] 启动失败: {exc}")
            self.modbus_server = None
            self.modbus_error = str(exc)
        self._refresh_modbus_status()

    def _toast(self, text: str, ms: int = 2200):
        if QtCore.QThread.currentThread() != self.thread():
            self.toast_sig.emit(text, ms)
            return
        self._show_toast(text, ms)

    @QtCore.pyqtSlot(str, int)
    def _show_toast(self, text: str, ms: int = 2200):
        self.status_label.setText(text)
        self.toast_timer.start(ms)

    # ---------- 摄像头控制 ----------
    def _start_camera(self):
        self.stop_camera()
        self.active_slots = []
        self._refresh_hik_devices()
        self._refresh_usb_indices()
        for slot in (1, 2):
            self._start_slot(slot)
        self._update_video_visibility()

    def _start_slot(self, slot: int):
        self._stop_slot(slot)
        source = self._slot_source(slot)
        if source == "hik":
            if not self.hik_devices:
                self._toast(f"未发现可用的海康相机（{slot}）")
                return
            idx = self.hik_indices.get(slot, 0)
            if idx >= len(self.hik_devices):
                idx = 0
                self.hik_indices[slot] = 0
            try:
                grabber = HikGrabber(self, device_index=int(idx))
            except Exception as exc:
                self._toast(f"海康{slot}启动失败：{exc}")
                return
        else:
            indices = list(self.usb_indices or [])
            if not indices:
                self._toast(f"未发现可用的 USB 摄像头（{slot}）")
                return
            idx = self.usb_indices_selected.get(slot, indices[0])
            if idx not in indices:
                idx = int(indices[0])
                self.usb_indices_selected[slot] = idx
            try:
                grabber = UsbGrabber(self, index=int(idx))
            except Exception as exc:
                self._toast(f"USB{slot} 启动失败：{exc}")
                return

        self.grabbers[slot] = grabber
        grabber.frameSignal.connect(lambda frame, s=slot: self.on_frame(s, frame))
        grabber.infoSignal.connect(lambda msg, s=slot: self.on_info(s, msg))
        grabber.errorSignal.connect(self._on_grabber_error)
        grabber.start()
        if slot not in self.active_slots:
            self.active_slots.append(slot)
        self._last_frame_ts[slot] = time.time()

    @QtCore.pyqtSlot(str)
    def _on_grabber_error(self, message: str):
        self._toast(message)

    def reopen_camera(self, slot: Optional[int] = None):
        if slot is None:
            self._start_camera()
        else:
            self._start_slot(int(slot))

    def _stop_slot(self, slot: int):
        grabber = self.grabbers.pop(slot, None)
        if grabber and getattr(grabber, "isRunning", lambda: False)():
            try:
                grabber.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            grabber.wait(1000)
        if slot in self.active_slots:
            try:
                self.active_slots.remove(slot)
            except ValueError:
                pass
        self._last_frame_ts[slot] = 0.0

    def stop_camera(self, slot: Optional[int] = None):
        if slot is None:
            for s in list(self.grabbers.keys()):
                self._stop_slot(int(s))
        else:
            self._stop_slot(int(slot))
        self._update_video_visibility()

    # ---------- 信号槽 ----------
    @QtCore.pyqtSlot(int, np.ndarray)
    def on_frame(self, cam_index: int, frame_bgr: np.ndarray):
        t = time.time()
        last_ts = self._last_paint_ts.get(cam_index, 0.0)
        if (t - last_ts) < (1.0 / UI_PAINT_FPS):
            return
        self._last_paint_ts[cam_index] = t
        self._last_frame_ts[cam_index] = t

        if frame_bgr is None or frame_bgr.size == 0:
            return

        self.last_frame_bgrs[cam_index] = frame_bgr

        frame_to_show = frame_bgr

        widgets = self.slot_widgets.get(cam_index, {})
        chk: QtWidgets.QCheckBox = widgets.get("chk_yolo")  # type: ignore
        use_yolo = bool(chk.isChecked()) if chk is not None else False
        if use_yolo:
            target_model = self.slot_model_paths.get(cam_index, self.default_model)
            if self._ensure_yolo_model(target_model):
                frame_to_show = self._apply_yolo(cam_index, frame_bgr)

        rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        lbl = self._video_label_for(cam_index)
        if not lbl:
            return
        scaled = pix.scaled(
            lbl.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation,
        )
        lbl.setPixmap(scaled)

    @QtCore.pyqtSlot(int, str)
    def on_info(self, cam_index: int, s: str):
        cam_label = self.lbl_cam1 if cam_index == 1 else self.lbl_cam2
        fps_label = self.lbl_fps1 if cam_index == 1 else self.lbl_fps2

        if s.startswith("[INFO]"):
            cam_label.setText(f"摄像头{cam_index}: " + s.replace("[INFO]", "").strip())
        elif s.startswith("[FPS]"):
            fps_label.setText(f"FPS{cam_index}: " + s.replace("[FPS]", "").strip())
        elif s.startswith("[HIK]") or s.startswith("[USB]"):
            self._toast(s)

    def _watchdog_tick(self):
        now = time.time()
        for slot in (1, 2):
            if slot not in self.active_slots:
                continue

            grabber = self.grabbers.get(slot)
            alive = bool(grabber and getattr(grabber, "isRunning", lambda: False)())
            last_frame_ts = self._last_frame_ts.get(slot, 0.0)
            stalled = (now - last_frame_ts) > 8.0 if last_frame_ts > 0 else False

            if alive and not stalled:
                continue

            last_restart = self._last_restart_ts.get(slot, 0.0)
            if (now - last_restart) < 5.0:
                continue

            reason = "线程已退出" if not alive else "自动重启"
            self._toast(f"摄像头{slot}: {reason}")
            self._last_restart_ts[slot] = now
            self._stop_slot(slot)
            QtCore.QTimer.singleShot(200, lambda s=slot: self._start_slot(s))

    def closeEvent(self, e):
        self._stop_modbus_server()
        self.stop_camera()
        if getattr(self, "_executor", None):
            self._executor.shutdown(wait=False)
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(resource_path(APP_ICON)))
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
