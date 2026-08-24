@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller
set "FFMPEG_BIN="
for /f "delims=" %%F in ('where ffmpeg 2^>nul') do if not defined FFMPEG_BIN set "FFMPEG_BIN=%%F"
if defined FFMPEG_BIN (
  python -m PyInstaller --noconfirm --clean --windowed --name MeteorStudio --hidden-import meteor_learning --hidden-import video_meteor --hidden-import alignment_workspace --hidden-import ptgui_pipeline --exclude-module torch --exclude-module torchvision --exclude-module transformers --exclude-module triton --exclude-module llvmlite --exclude-module numba --exclude-module onnx --exclude-module onnxruntime --exclude-module matplotlib --exclude-module timm --exclude-module tokenizers --add-data "meteor_ranker.json;." --add-data "candidate_dataset.npz;." --add-binary "%FFMPEG_BIN%;." meteor_composer.py
) else (
  echo 警告：未找到 FFmpeg，视频导出将要求目标电脑自行安装 FFmpeg。
  python -m PyInstaller --noconfirm --clean --windowed --name MeteorStudio --hidden-import meteor_learning --hidden-import video_meteor --hidden-import alignment_workspace --hidden-import ptgui_pipeline --exclude-module torch --exclude-module torchvision --exclude-module transformers --exclude-module triton --exclude-module llvmlite --exclude-module numba --exclude-module onnx --exclude-module onnxruntime --exclude-module matplotlib --exclude-module timm --exclude-module tokenizers --add-data "meteor_ranker.json;." --add-data "candidate_dataset.npz;." meteor_composer.py
)
pause
