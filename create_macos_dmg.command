#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

version="${1:-0.1.0}"
app_path="dist/MeteorStudio.app"
architecture="$(uname -m)"
dmg_path="dist/MeteorStudio-${version}-${architecture}.dmg"

if [[ ! -d "$app_path" ]]; then
  echo "找不到 $app_path，请先运行 ./build_macos.command"
  exit 1
fi

staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/meteor-studio-dmg.XXXXXX")"
trap 'rm -rf "$staging_dir"' EXIT

ditto "$app_path" "$staging_dir/MeteorStudio.app"
ln -s /Applications "$staging_dir/Applications"
rm -f "$dmg_path"

hdiutil create \
  -volname "MeteorStudio" \
  -srcfolder "$staging_dir" \
  -ov \
  -format UDZO \
  "$dmg_path"

echo "完成：$dmg_path"

