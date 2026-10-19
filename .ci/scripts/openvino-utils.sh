#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export_openvino_lib_path() {
  # The OpenVINO wheel only ships a versioned libopenvino_c.so, which the backend dlopens.
  OPENVINO_LIB_PATH=$(${PYTHON_EXECUTABLE:-python} -c "import openvino, pathlib; print(next(pathlib.Path(openvino.__file__).with_name('libs').glob('libopenvino_c.so*')))")
  export OPENVINO_LIB_PATH
}
