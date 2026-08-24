# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

import copy
import gc
from typing import final, List

import torch

from executorch.backends.openvino._passes import DecomposeFloorDividePass

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass
from openvino.frontend.pytorch.torchdynamo.compile import (  # type: ignore[import-untyped]
    openvino_compile,
)


@final
class OpenvinoBackend(BackendDetails):

    @classmethod
    def copy_exported_program_for_preprocess(
        cls,
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> ExportedProgram:
        """Isolate the graph ``preprocess`` mutates without copying weights.

        The default deepcopies the whole program, weights included. The passes
        below only rewrite graph nodes, so an independent ``fx.Graph`` over the
        same root module is enough; ``GraphModule(root, graph)`` re-registers
        parameters by reference.
        """
        graph_module = edge_program.graph_module
        isolated_root = torch.fx.GraphModule(
            graph_module, copy.deepcopy(graph_module.graph)
        )
        return ExportedProgram(
            root=isolated_root,
            graph=isolated_root.graph,
            graph_signature=copy.deepcopy(edge_program.graph_signature),
            state_dict=edge_program.state_dict,
            range_constraints=copy.deepcopy(edge_program.range_constraints),
            module_call_graph=copy.deepcopy(edge_program.module_call_graph),
            example_inputs=edge_program.example_inputs,
            constants=edge_program.constants,
            verifiers=[edge_program.verifier],
        )

    @classmethod
    def preprocess(
        cls, edge_program: ExportedProgram, module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:
        """
        Preprocesses the exported program and compiles it for the OpenVINO backend.

        Args:
            edge_program (ExportedProgram): The exported program representing the model.
            module_compile_spec (List[CompileSpec]): A list of compile specifications for the OpenVINO backend.

        Returns:
            PreprocessResult: The result of preprocessing, including the compiled model bytes.
        """
        for pass_cls in [DimOrderOpsRevertPass, DecomposeFloorDividePass]:
            result = pass_cls()(edge_program.graph_module)
            if result and result.graph_module:
                edge_program._graph_module = result.graph_module

        input_names = edge_program.graph_signature.user_inputs
        args = []
        for node in edge_program.graph.nodes:
            if node.target in input_names:
                args.append(node.meta["val"])

        compile_options = {}
        for spec in module_compile_spec:
            compile_options[spec.key] = spec.value.decode()

        compiled = openvino_compile(
            edge_program.module(), *args, options=compile_options
        )
        model_bytes = compiled.export_model()

        # Drop the compiled model's copy of the weights; only the bytes are
        # needed from here.
        del compiled
        gc.collect()

        return PreprocessResult(processed_bytes=model_bytes.getvalue())
