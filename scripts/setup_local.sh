#!/bin/bash
# 코랩이 아닌 컴퓨터에서 실험할 FedACG 폴더를 만든다 (노트북 1~2번 셀과 같은 일 + Mac MPS 사용).
#   bash scripts/setup_local.sh ~/fedacg-local
# 이미 있으면 지우고 새로 만든다. data/ 폴더(CIFAR-10)는 남겨 둔다.
set -euo pipefail
DEST=${1:?사용법: bash scripts/setup_local.sh <FedACG를 만들 폴더>}
T=$(cd "$(dirname "$0")/.." && pwd)

if [ -d "$DEST/data" ]; then mv "$DEST/data" "$DEST.data.keep"; fi
rm -rf "$DEST"
git clone -q https://github.com/geehokim/FedACG.git "$DEST"
if [ -d "$DEST.data.keep" ]; then mv "$DEST.data.keep" "$DEST/data"; fi

cd "$DEST"
for pkg in attacks metrics defenses resume; do
  [ -d "$T/$pkg" ] && cp -R "$T/$pkg" .
done
for g in "$T"/configs/*; do
  n=$(basename "$g")
  if [ -d "$g" ] && [ "$n" != experiments ]; then cp -R "$g" configs/; fi   # 끝에 / 없이: 폴더째 복사
done
for p in fedacg_integration metrics_integration defense_integration colab_memory resume_integration mps_device; do
  git apply "$T/integration/$p.patch" && echo "applied $p"
done
echo "준비 끝: $DEST (CIFAR-10은 $DEST/data, 없으면 첫 실행 때 받는다)"
