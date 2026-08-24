#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
version="${1:-0.1.0}"

chmod +x build_macos.command create_macos_dmg.command
./build_macos.command
./create_macos_dmg.command "$version"

