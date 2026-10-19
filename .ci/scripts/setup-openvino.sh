#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/openvino-utils.sh"

# Parse arguments
USE_NIGHTLY=false
for arg in "$@"; do
  case $arg in
    --nightly) USE_NIGHTLY=true ;;
  esac
done

OPENVINO_VERSION="2026.4.1"
OPENVINO_NIGHTLY_VERSION="2026.5.0.dev20261019"

if [ "${USE_NIGHTLY}" = true ]; then
  echo "Using OpenVINO nightly build: ${OPENVINO_NIGHTLY_VERSION}"
  pip install --pre "openvino==${OPENVINO_NIGHTLY_VERSION}" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
else
  echo "Using OpenVINO stable release: ${OPENVINO_VERSION}"
  pip install "openvino==${OPENVINO_VERSION}"
fi

export_openvino_lib_path

pip install -r backends/openvino/requirements.txt
pushd backends/openvino/scripts
./openvino_build.sh --enable_python
popd
