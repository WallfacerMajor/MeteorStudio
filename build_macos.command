#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "找不到 python3。推荐先执行：brew install python@3.12"
  exit 1
fi

venv_dir=".venv-build-macos"
python3 -m venv "$venv_dir"
python_bin="$venv_dir/bin/python"

"$python_bin" -m pip install --upgrade pip wheel
"$python_bin" -m pip install -r requirements.txt pyinstaller

args=(
  --noconfirm
  --clean
  --windowed
  --name MeteorStudio
  --osx-bundle-identifier com.wallfacemajor.meteorstudio
  --hidden-import meteor_learning
  --hidden-import video_meteor
  --hidden-import alignment_workspace
  --hidden-import ptgui_pipeline
  --hidden-import gui_interaction_smoke
  --exclude-module torch
  --exclude-module torchvision
  --exclude-module transformers
  --exclude-module triton
  --exclude-module llvmlite
  --exclude-module numba
  --exclude-module onnx
  --exclude-module onnxruntime
  --exclude-module matplotlib
  --exclude-module timm
  --exclude-module tokenizers
  --add-data "meteor_ranker.json:."
  --add-data "candidate_dataset.npz:."
)

if command -v ffmpeg >/dev/null 2>&1; then
  args+=(--add-binary "$(command -v ffmpeg):.")
else
  echo "警告：未找到 FFmpeg。图片功能可用，视频导出需要另行安装 FFmpeg。"
fi

"$python_bin" -m PyInstaller "${args[@]}" meteor_composer.py

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - dist/MeteorStudio.app
fi

echo "完成：dist/MeteorStudio.app ($(uname -m))"
