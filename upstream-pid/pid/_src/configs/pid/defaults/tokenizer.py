# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from pid._src.tokenizers.flux2_vae import Flux2VAEConfig
from pid._src.tokenizers.flux_vae import FluxVAEConfig, SD3VAEConfig


def register_tokenizer():
    cs = ConfigStore.instance()
    cs.store(group="tokenizer", package="model.config.tokenizer", name="flux_vae_tokenizer", node=FluxVAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="sd3_vae_tokenizer", node=SD3VAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="flux2_vae_tokenizer", node=Flux2VAEConfig)
