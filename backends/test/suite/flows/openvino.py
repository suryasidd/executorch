# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable

from executorch.backends.openvino.test.tester import (
    Quantize as OpenvinoQuantize,
    Tester as OpenvinoTester,
)
from executorch.backends.test.harness.stages import Quantize
from executorch.backends.test.suite.flow import TestFlow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _create_openvino_flow_base(
    name: str, quantize_stage_factory: Callable[..., Quantize] | None = None
) -> TestFlow:
    return TestFlow(
        name,
        backend="openvino",
        tester_factory=OpenvinoTester,
        quantize=quantize_stage_factory is not None,
        quantize_stage_factory=quantize_stage_factory,
    )


def _create_openvino_flow() -> TestFlow:
    return _create_openvino_flow_base("openvino")


def _create_openvino_int8_flow() -> TestFlow:
    """
    INT8 quantization flow for OpenVINO.
    Uses post-training quantization with calibration.
    """
    def create_quantize_stage() -> Quantize:
        return OpenvinoQuantize(
            calibrate=True,
        )

    return _create_openvino_flow_base("openvino_int8", create_quantize_stage)


OPENVINO_TEST_FLOW = _create_openvino_flow()
OPENVINO_INT8_TEST_FLOW = _create_openvino_int8_flow()
