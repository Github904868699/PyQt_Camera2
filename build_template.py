from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


# =========================
# 0) 配置区：你只需要改这里
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

# 你的核心模块：先用 Nuitka 编译成 .pyd
NUITKA_TARGET = PROJECT_ROOT / "camera_core.py"

# 你的入口脚本：PyInstaller 打包它（入口里 import camera_core / 或调用你的主逻辑）
ENTRY_SCRIPT = PROJECT_ROOT / "Camera_entry.py"

APP_NAME = "Camera"
ICON_PATH = PROJECT_ROOT / "Camera.ico"          # 没有就设为 None
EXTRA_DLL_DIRS = [
    PROJECT_ROOT / "Win64_x64",                  # 你的海康/第三方 dll 目录
]

# 输出目录
NUITKA_OUT_DIR = PROJECT_ROOT / "build_nuitka"
PYI_DIST_DIR = PROJECT_ROOT / "dist"
PYI_WORK_DIR = PROJECT_ROOT / "build"

# 打包形式
ENABLE_CONSOLE = False      # True=带控制台（排错），False=无控制台
ENABLE_ONEFILE = False      # True=onefile，False=onedir（建议先 onedir 稳定再 onefile）

# 关键：onnxruntime 防炸策略
ENABLE_ORT_HOOK = True
ORT_EARLY_IMPORT = True     # 早预加载 onnxruntime_pybind11_state（强烈建议 True）


# =========================
# 1) 工具函数
# =========================
def run(cmd: List[str], cwd: Path | None = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), check=True)


def clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def find_site_pkg_module_path(modname: str) -> Path:
    """返回某个包在当前 python 环境下的安装路径（目录）"""
    import importlib
    m = importlib.import_module(modname)
    # package: __file__ 在包内某个 __init__.py 上
    return Path(m.__file__).resolve().parent


def glob_add_binaries_from_dir(src_dir: Path, dst_rel: str) -> List[str]:
    """把目录下常见二进制收集为 PyInstaller --add-binary 形式: 'src;dst'"""
    if not src_dir.exists():
        return []
    bins = []
    for ext in (".dll", ".pyd", ".so", ".dylib", ".exe"):
        for f in src_dir.glob(f"*{ext}"):
            bins.append(f"{f};{dst_rel}")
    return bins


# =========================
# 2) Nuitka 编译：生成 .pyd
# =========================
def run_nuitka_compile() -> Path:
    if not NUITKA_TARGET.exists():
        raise FileNotFoundError(f"Nuitka target not found: {NUITKA_TARGET}")

    NUITKA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nuitka",
        "--mode=module",
        "--msvc=latest",
        f"--output-dir={NUITKA_OUT_DIR}",
        "--no-pyi-file",
        "--assume-yes-for-downloads",
        str(NUITKA_TARGET),
    ]
    run(cmd, cwd=PROJECT_ROOT)

    # 找生成的 pyd
    candidates = list(NUITKA_OUT_DIR.rglob("*.pyd"))
    if not candidates:
        raise RuntimeError(f"No .pyd produced in {NUITKA_OUT_DIR}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    built = candidates[0]

    # 拷贝到项目根目录，保证 PyInstaller/运行时 import 稳
    dst = PROJECT_ROOT / built.name
    shutil.copy2(built, dst)
    print("[Nuitka] Built:", built)
    print("[Nuitka] Copied:", dst)
    return dst


# =========================
# 3) 生成 runtime-hook：DLL 搜索路径 + ORT 早预加载
# =========================
def write_runtime_hook_onnxruntime(hook_path: Path) -> None:
    content = r"""# Auto-generated PyInstaller runtime hook
import os
import sys
import traceback
from pathlib import Path

base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

# 兜底：OpenMP 冲突（保守开，优先保证能跑）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def add_dir(p: Path):
    try:
        if p and p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# onedir: base 往往是 dist\\App\\_internal
# onefile: base 是临时解压目录
cands = [
    base,
    base / "onnxruntime",
    base / "onnxruntime" / "capi",
    base / "_internal",
    base / "_internal" / "onnxruntime",
    base / "_internal" / "onnxruntime" / "capi",
    base.parent,
    base.parent / "onnxruntime",
    base.parent / "onnxruntime" / "capi",
    base.parent / "_internal",
    base.parent / "_internal" / "onnxruntime",
    base.parent / "_internal" / "onnxruntime" / "capi",
]

for p in cands:
    add_dir(p)

# 关键：尽早把 ORT 的 pybind 状态加载进来（避免后续 DLL 污染导致 1114）
try:
    import onnxruntime.capi.onnxruntime_pybind11_state  # noqa: F401
except Exception:
    # 无控制台时，把错误写到文件，便于定位
    try:
        log_dir = base.parent if base.name.lower() == "_internal" else base
        (log_dir / "onnxruntime_import_error.log").write_text(traceback.format_exc(), encoding="utf-8")
    except Exception:
        pass
"""
    hook_path.write_text(content, encoding="utf-8")
    print("[Hook] Written:", hook_path)


# =========================
# 4) PyInstaller 打包：强制收集 onnxruntime capi 二进制
# =========================
def build_pyinstaller_cmd(runtime_hook: Path | None) -> List[str]:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        f"--distpath={PYI_DIST_DIR}",
        f"--workpath={PYI_WORK_DIR}",
    ]

    if ENABLE_CONSOLE:
        cmd += ["--console"]
    else:
        cmd += ["--noconsole"]

    if ENABLE_ONEFILE:
        cmd += ["--onefile"]
    else:
        cmd += ["--onedir"]

    if ICON_PATH and ICON_PATH.exists():
        cmd += ["--icon", str(ICON_PATH)]
        cmd += ["--add-data", f"{ICON_PATH};."]

    if runtime_hook is not None:
        cmd += ["--runtime-hook", str(runtime_hook)]

    # ---- 关键：Nuitka 后 PyInstaller 不易分析 imports，hidden-import 要保守一些
    # 你可按项目增删
    hidden_imports = [
        "numpy",
        "cv2",
        "onnxruntime",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
    ]
    for hi in hidden_imports:
        cmd += ["--hidden-import", hi]

    # ---- 强制收集 onnxruntime 包（python 文件+metadata）
    cmd += ["--collect-all", "onnxruntime"]
    cmd += ["--collect-all", "cv2"]
    cmd += ["--collect-all", "PyQt5"]

    # ---- 最关键：把 onnxruntime\\capi 下的 DLL + PYD 显式打进包里，并保持目录结构
    ort_dir = find_site_pkg_module_path("onnxruntime")
    ort_capi = ort_dir / "capi"
    for item in glob_add_binaries_from_dir(ort_capi, r"_internal\onnxruntime\capi"):
        cmd += ["--add-binary", item]

    # ---- 你的第三方 DLL 目录（海康等）
    for d in EXTRA_DLL_DIRS:
        for item in glob_add_binaries_from_dir(d, "Win64_x64"):
            cmd += ["--add-binary", item]

    # 入口
    cmd += [str(ENTRY_SCRIPT)]
    return cmd


def run_pyinstaller() -> None:
    if not ENTRY_SCRIPT.exists():
        raise FileNotFoundError(f"Entry script not found: {ENTRY_SCRIPT}")

    runtime_hook = None
    if ENABLE_ORT_HOOK:
        runtime_hook = PROJECT_ROOT / "rth_onnxruntime.py"
        write_runtime_hook_onnxruntime(runtime_hook)

    cmd = build_pyinstaller_cmd(runtime_hook)
    run(cmd, cwd=PROJECT_ROOT)


# =========================
# 5) 主流程
# =========================
def main() -> None:
    print("=== Build Start ===")
    print("Project:", PROJECT_ROOT)
    print("Python :", sys.executable)

    # 建议：每次干净构建
    clean_dir(PYI_DIST_DIR / APP_NAME)
    clean_dir(PYI_WORK_DIR)

    # 先 Nuitka 再 PyInstaller
    run_nuitka_compile()
    run_pyinstaller()

    print("=== Build Done ===")
    print("Output:", PYI_DIST_DIR / APP_NAME)


if __name__ == "__main__":
    main()
