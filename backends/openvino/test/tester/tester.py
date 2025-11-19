# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Sequence, Tuple

import executorch
import executorch.backends.test.harness.stages as BaseStages
import torch
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.quantizer.quantizer import OpenVINOQuantizer
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import Partitioner


class Export(BaseStages.Export):
    pass


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        quantizer=None,
        quantization_config=None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
    ):
        super().__init__(
            quantizer=quantizer or OpenVINOQuantizer(),
            quantization_config=quantization_config,
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )


class RunPasses(BaseStages.RunPasses):
    pass


class ToEdge(BaseStages.ToEdge):
    def __init__(self, edge_compile_config: Optional[EdgeCompileConfig] = None):
        super().__init__(edge_compile_config or EdgeCompileConfig(_check_ir_validity=False))


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        # If no compile specs provided, default to CPU
        if compile_specs is None:
            compile_specs = [CompileSpec("device", b"CPU")]

        # If no partitioners provided, use OpenvinoPartitioner with compile specs
        if partitioners is None:
            partitioners = [OpenvinoPartitioner(compile_specs)]

        super().__init__(
            default_partitioner_cls=None,  # We're providing explicit partitioners
            partitioners=partitioners,
            edge_compile_config=edge_compile_config or EdgeCompileConfig(_check_ir_validity=False),
        )


class Partition(BaseStages.Partition):
    def __init__(
        self,
        partitioner: Optional[Partitioner] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        # If no compile specs provided, default to CPU
        if compile_specs is None:
            compile_specs = [CompileSpec("device", b"CPU")]

        super().__init__(
            partitioner=partitioner or OpenvinoPartitioner(compile_specs),
        )


class Serialize(BaseStages.Serialize):
    pass


class ToExecutorch(BaseStages.ToExecutorch):
    pass


class Tester(TesterBase):
    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        compile_specs: Optional[List[CompileSpec]] = None,
    ):
        # Store compile specs for use in stages
        self.compile_specs = compile_specs or [CompileSpec("device", b"CPU")]

        # Specialize for OpenVINO
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.EXPORT: Export,
                StageType.PARTITION: lambda: Partition(
                    compile_specs=self.compile_specs
                ),
                StageType.QUANTIZE: Quantize,
                StageType.RUN_PASSES: RunPasses,
                StageType.TO_EDGE: ToEdge,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: lambda: ToEdgeTransformAndLower(
                    compile_specs=self.compile_specs
                ),
                StageType.SERIALIZE: Serialize,
                StageType.TO_EXECUTORCH: ToExecutorch,
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
