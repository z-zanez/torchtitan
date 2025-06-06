# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance

import torch
import torch.nn as nn

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class Float8Converter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = False

        float8_config = job_config.float8
        if not has_cuda_capability(8, 9):
            logger.warning(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
            )
            return
        try:
            from torchao.float8 import Float8LinearConfig
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        if float8_config.recipe_name is not None and not hasattr(
            Float8LinearConfig, "from_recipe_name"
        ):
            logger.warning(
                "Failed to swap to Float8Linear with recipe lookup because the torchao version "
                "is too old, please install torchao v0.9.0 or later and try again",
            )
            return

        self.enabled = True
        self.filter_fqns = float8_config.filter_fqns

        if float8_config.recipe_name is not None:
            assert (
                not float8_config.enable_fsdp_float8_all_gather
            ), "using `float8_config.enable_fsdp_float8_all_gather` together with `float8_config.recipe_name` is not supported"
            assert (
                not float8_config.force_recompute_fp8_weight_in_bwd
            ), "using `float8_config.force_recompute_fp8_weight_in_bwd` together with `float8_config.recipe_name` is not supported"
            self.config = Float8LinearConfig.from_recipe_name(float8_config.recipe_name)
            self.precompute_scale = False
            logger.info(
                f"Float8 training active with recipe {float8_config.recipe_name}"
            )

            # short-term solution for https://github.com/pytorch/pytorch/issues/150859
            if float8_config.recipe_name == "rowwise":
                torch._inductor.config.emulate_precision_casts = True
                logger.debug(
                    "Set torch._inductor.config.emulate_precision_casts to True"
                )

        else:
            # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
            enable_fsdp_float8_all_gather = (
                parallel_dims.dp_shard_enabled
                and float8_config.enable_fsdp_float8_all_gather
            )
            self.config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                force_recompute_fp8_weight_in_bwd=float8_config.force_recompute_fp8_weight_in_bwd,
            )
            # for precompute_float8_dynamic_scale_for_fsdp
            self.precompute_scale = (
                enable_fsdp_float8_all_gather
                and float8_config.precompute_float8_dynamic_scale_for_fsdp
            )
            logger.info("Float8 tensorwise scaled training active")

    def convert(self, model: nn.Module):
        return self.convert_to_float8_training(model)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        return self.precompute_float8_dynamic_scale_for_fsdp(model)

    def convert_to_float8_training(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=self._module_filter_fn,
        )
        logger.info(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def _module_filter_fn(self, mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False

        # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
        dims_multiples_of_16 = (
            mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0
        )

        # If the fqn matches any filtered fqn, then we should not convert this module.
        is_filtered_fqn = any(filtered_fqn in fqn for filtered_fqn in self.filter_fqns)

        return dims_multiples_of_16 and not is_filtered_fqn

    def precompute_float8_dynamic_scale_for_fsdp(
        self, model: nn.Module | list[nn.Module]
    ):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)


register_model_converter(Float8Converter, "float8")
