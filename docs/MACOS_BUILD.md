# macOS 构建与 DMG 制作

Windows 不能直接生成可运行的 macOS 应用。必须在真实 Mac 或 macOS 构建机上执行以下步骤。

## 1. 支持的架构

- Apple Silicon Mac 构建得到 `arm64` 应用。
- Intel Mac 构建得到 `x86_64` 应用。
- 默认脚本构建当前 Mac 的原生架构，不会伪造 Universal 2。

如果要发布 Universal 2，推荐分别在 Apple Silicon 和 Intel 环境构建两个版本并分别发布。也可以使用 Universal 2 Python 和所有依赖的 Universal 2 wheel，但必须逐个验证 OpenCV、NumPy、SciPy 和 scikit-learn 的架构。

## 2. 安装基础环境

推荐先安装 Homebrew，然后执行：

```bash
brew install python@3.12 ffmpeg
```

再安装：

- PTGui Pro：通常位于 `/Applications/PTGui Pro.app`。
- Siril：通常位于 `/Applications/Siril.app`；也可以通过 Homebrew 安装提供 `siril-cli` 的版本。

MeteorStudio 会自动尝试：

- `/Applications/PTGui Pro.app/Contents/MacOS/PTGui Pro`
- `/Applications/Siril.app/Contents/MacOS/siril-cli`
- `PATH` 中的 `siril-cli` 和 `ffmpeg`

找不到时可在星空对齐窗口中手动选择。

## 3. 获取源码

```bash
git clone https://github.com/WallfacerMajor/MeteorStudio.git
cd MeteorStudio
```

## 4. 一键生成 APP

```bash
chmod +x build_macos.command
./build_macos.command
```

脚本会创建隔离的 `.venv-build-macos`，安装依赖并生成：

```text
dist/MeteorStudio.app
```

如果系统存在 FFmpeg，脚本会把它放入应用包；PTGui 和 Siril因许可证及体积原因不会被打包。

## 5. 一键生成 DMG

```bash
chmod +x build_macos_dmg.command create_macos_dmg.command
./build_macos_dmg.command 0.1.0
```

结果示例：

```text
dist/MeteorStudio-0.1.0-arm64.dmg
```

也可以在 APP 已经存在时单独运行：

```bash
./create_macos_dmg.command 0.1.0
```

## 6. 未签名版本的打开方式

未签名应用适合自己测试。首次打开时：

1. 在 Finder 中右键 `MeteorStudio.app`，选择“打开”。
2. 再次确认“打开”。

如果应用是你本人从本仓库构建并确认可信，也可以执行：

```bash
xattr -dr com.apple.quarantine dist/MeteorStudio.app
open dist/MeteorStudio.app
```

不要对来源不明的应用移除隔离属性。

## 7. Developer ID 签名与公证

面向其他用户发布时，建议准备 Apple Developer Program 的 Developer ID Application 证书。

签名 APP：

```bash
codesign --deep --force --options runtime --timestamp \
  --sign "Developer ID Application: YOUR NAME (TEAMID)" \
  dist/MeteorStudio.app

codesign --verify --deep --strict --verbose=2 dist/MeteorStudio.app
spctl --assess --type execute --verbose=2 dist/MeteorStudio.app
```

保存公证凭据：

```bash
xcrun notarytool store-credentials "MeteorStudio-notary" \
  --apple-id "YOUR_APPLE_ID" \
  --team-id "TEAMID" \
  --password "APP_SPECIFIC_PASSWORD"
```

提交 APP 公证：

```bash
ditto -c -k --keepParent dist/MeteorStudio.app dist/MeteorStudio-notary.zip
xcrun notarytool submit dist/MeteorStudio-notary.zip \
  --keychain-profile "MeteorStudio-notary" --wait
xcrun stapler staple dist/MeteorStudio.app
```

完成后再运行 `create_macos_dmg.command`，并签名 DMG：

```bash
codesign --force --timestamp \
  --sign "Developer ID Application: YOUR NAME (TEAMID)" \
  dist/MeteorStudio-0.1.0-arm64.dmg
```

Apple 不允许在 Windows 上完成有效的 macOS Developer ID 签名和公证。

## 8. 构建验证

```bash
open dist/MeteorStudio.app
file dist/MeteorStudio.app/Contents/MacOS/MeteorStudio
codesign -dv --verbose=4 dist/MeteorStudio.app 2>&1 || true
```

至少验证：程序启动、图片读取、项目保存、Siril 星点检测、PTGui 单张工程生成以及 FFmpeg 视频导出。

