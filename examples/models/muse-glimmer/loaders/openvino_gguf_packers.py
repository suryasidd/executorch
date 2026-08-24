# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Repack GGUF weights into the portable int4/int8 forms the OpenVINO backend
understands.

There is no frontend rule for ``torchao::dequantize_gguf``, so Q4_K becomes
``ExportableInt4Tensor`` and Q5_K/Q6_K become ``IntxUnpackedToInt8Tensor`` --
the same split llama.cpp's OpenVINO backend uses. Dequantizing to bf16 instead
would roughly double those 53 tensors (~7.9 GB int8 vs ~13.8 GB bf16).
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn


def convert_quantized_tensors_for_openvino(
    model: nn.Module, activation_dtype: torch.dtype = torch.bfloat16
) -> dict[str, int]:
    """Rewrite every ``ExportableGGUFTensor`` parameter in ``model`` in place.

    Returns a summary count per action, for logging.
    """
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor

    stats = {"q4_k_to_int4": 0, "to_int8": 0, "dequantized": 0, "untouched": 0}
    converted = 0

    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            value = param.data if isinstance(param, nn.Parameter) else param
            if not isinstance(value, ExportableGGUFTensor):
                stats["untouched"] += 1
                continue

            if value.ggml_type == "q4_k":
                # Scales stay bf16, not GGUF's native fp16: the frontend's
                # dequantize_int4_tensor falls back to the scale dtype for its
                # output, and fp16 there fails the following bf16 MatMul.
                new_value = value.to_exportable_int4_tensor()
                stats["q4_k_to_int4"] += 1
            else:
                # Q5_K/Q6_K need more than 4 bits; int8 keeps them compressed.
                try:
                    new_value = value.to_intx_unpacked_to_int8_tensor()
                    stats["to_int8"] += 1
                except Exception:
                    new_value = value.dequantize(activation_dtype)
                    stats["dequantized"] += 1

            setattr(module, name, nn.Parameter(new_value, requires_grad=False))

            # Release the source tensor now; otherwise every original and its
            # repacked copy stay resident until the loop ends. Collecting every
            # 32nd keeps the cost off the wall clock.
            del param, value, new_value
            converted += 1
            if converted % 32 == 0:
                gc.collect()

    gc.collect()
    return stats
