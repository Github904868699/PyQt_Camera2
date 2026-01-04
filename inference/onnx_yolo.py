# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_ORT_SPEC = importlib.util.find_spec("onnxruntime")
if _ORT_SPEC is None:
    ort = None  # type: ignore
    ORT_AVAILABLE = False
    ORT_IMPORT_ERROR = ImportError("onnxruntime 未安装")
else:
    import onnxruntime as ort

    ORT_AVAILABLE = True
    ORT_IMPORT_ERROR = None


def list_onnx_models(base_dir: str | None = None) -> List[str]:
    try:
        root = Path(base_dir) if base_dir else Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    except Exception:
        root = Path(__file__).resolve().parent.parent

    models: List[str] = []
    try:
        for p in sorted(root.glob("*.onnx")):
            if p.is_file():
                models.append(str(p))
    except Exception:
        pass

    return models


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep


class OnnxYoloModel:
    def __init__(self, model_path: str):
        if not ORT_AVAILABLE or ort is None:
            raise RuntimeError(f"onnxruntime 未就绪: {ORT_IMPORT_ERROR}")

        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.input_h, self.input_w = self._resolve_input_shape(input_meta.shape)
        self.names = self._load_names(model_path)

    @staticmethod
    def _resolve_input_shape(shape) -> Tuple[int, int]:
        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
            h = shape[2] if isinstance(shape[2], int) and shape[2] else 640
            w = shape[3] if isinstance(shape[3], int) and shape[3] else 640
            return int(h), int(w)
        return 640, 640

    @staticmethod
    def _load_names(model_path: str) -> Dict[int, str]:
        names_path = Path(model_path).with_suffix(".names")
        if not names_path.exists():
            return {}
        try:
            lines = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines()]
        except Exception:
            return {}
        return {idx: name for idx, name in enumerate(lines) if name}

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        h, w = img.shape[:2]
        ratio = min(self.input_w / w, self.input_h / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.input_w - new_w
        pad_h = self.input_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, ratio, float(left), float(top)

    def label_for(self, cls_id: int) -> str:
        return self.names.get(int(cls_id), str(int(cls_id)))

    def predict(self, frame_bgr: np.ndarray, conf_thres: float, iou_thres: float = 0.45):
        img, ratio, pad_x, pad_y = self._letterbox(frame_bgr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: img})
        preds = outputs[0]

        if preds.ndim == 3:
            if preds.shape[1] < preds.shape[2]:
                preds = preds.transpose(0, 2, 1)
            preds = preds[0]
        elif preds.ndim == 2:
            preds = preds
        else:
            preds = preds.reshape(-1, preds.shape[-1])

        if preds.shape[-1] == 6:
            boxes = preds[:, :4]
            scores = preds[:, 4]
            cls_ids = preds[:, 5].astype(int)
        else:
            boxes = preds[:, :4]
            scores_all = preds[:, 4:]
            cls_ids = np.argmax(scores_all, axis=1)
            scores = scores_all[np.arange(scores_all.shape[0]), cls_ids]

        mask = scores >= float(conf_thres)
        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if boxes.size == 0:
            return []

        x_c, y_c, w, h = boxes.T
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        x1 = np.clip(x1, 0, frame_bgr.shape[1] - 1)
        y1 = np.clip(y1, 0, frame_bgr.shape[0] - 1)
        x2 = np.clip(x2, 0, frame_bgr.shape[1] - 1)
        y2 = np.clip(y2, 0, frame_bgr.shape[0] - 1)

        final_boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms_boxes(final_boxes, scores, iou_thres)
        results = []
        for idx in keep:
            results.append(
                {
                    "bbox": final_boxes[idx],
                    "score": float(scores[idx]),
                    "cls_id": int(cls_ids[idx]),
                }
            )
        return results
