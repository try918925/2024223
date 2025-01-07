# -*- mode: python ; coding: utf-8 -*-
import os
def collect_files(source_dir, target_dir):
    collected = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file).replace("\\", "/")
            rel_path = os.path.relpath(full_path, source_dir).replace("\\", "/")
            target_path = os.path.join(target_dir, rel_path).replace("\\", "/")
            target_path = os.path.dirname(target_path)
            not_need_path = target_path.split("/")[-1]
            if not not_need_path in ["__pycache__"]:
                collected.append((full_path, target_path))
    return collected

datas = [
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/onnx', 'onnx'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/onnx-1.15.0.dist-info', 'onnx-1.15.0.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/paddle', 'paddle'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/paddlepaddle_gpu-2.6.2.dist-info', 'paddlepaddle_gpu-2.6.2.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/paddle_bfloat-0.1.7.dist-info', 'paddle_bfloat-0.1.7.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torch', 'torch'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torch-1.13.1+cu117.dist-info', 'torch-1.13.1+cu117.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torchaudio', 'torchaudio'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torchaudio-0.13.1+cu117.dist-info', 'torchaudio-0.13.1+cu117.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torchgen', 'torchgen'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torchvision', 'torchvision'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/torchvision-0.14.1+cu117.dist-info', 'torchvision-0.14.1+cu117.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/ultralytics', 'ultralytics'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/ultralytics-8.3.34.dist-info', 'ultralytics-8.3.34.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/ultralytics_thop-2.0.12.dist-info', 'ultralytics_thop-2.0.12.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/scipy', 'scipy'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/scipy-1.10.1.dist-info', 'scipy-1.10.1.dist-info'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/scipy.libs', 'scipy.libs'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/tensorrt', 'tensorrt'),
    ('C:/Users/Install/.conda/envs/p8_t113_c118/Lib/site-packages/tensorrt-8.6.0.dist-info', 'tensorrt-8.6.0.dist-info'),
    ]

datas += collect_files("C:/Users/Install/Desktop/pack_install/output", "output")
datas += collect_files("C:/Users/Install/Desktop/pack_install/output_car", "output_car")
datas += collect_files("C:/Users/Install/Desktop/pack_install/ppocr", "ppocr")
datas += collect_files("C:/Users/Install/Desktop/pack_install/ppocr_car", "ppocr_car")
datas += collect_files("C:/Users/Install/Desktop/pack_install/algorithms", "algorithms")
datas += collect_files("C:/Users/Install/Desktop/pack_install/fonts", "fonts")
datas += collect_files("C:/Users/Install/Desktop/pack_install/tools", "tools")
datas += collect_files("C:/Users/Install/Desktop/pack_install/configs", "configs")
datas += collect_files("C:/Users/Install/Desktop/pack_install/config_car", "config_car")
datas += collect_files("C:/Users/Install/Desktop/pack_install/config", "config")
datas += collect_files("C:/Users/Install/Desktop/pack_install/models", "models")
datas += collect_files("C:/Users/Install/Desktop/pack_install/weights", "weights")
datas += collect_files("C:/Users/Install/Desktop/pack_install/utils", "utils")

a = Analysis(
    ['V0.2.0_version_iteration.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
    "algorithms.models_sx",
    "algorithms.utils_sx",
    "algorithms.data",
    "algorithms.runs",
    "ppocr.data",
    "ppocr.modeling",
    "ppocr.postprocess",
    "ppocr.utils",
    "ppocr_car.data",
    "ppocr_car.modeling",
    "ppocr_car.postprocess",
    "ppocr_car.utils",
    "tools.end2end",
    "tools.infer",
    "utils.aws",
    "utils.google_app_engine",
    "utils.wandb_logging",
    "output.db_mv3",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
  )
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='tianjingang',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)