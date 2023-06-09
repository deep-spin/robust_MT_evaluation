# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Bottleneck Layer
==============
    Bottleneck module to be used with customised features
"""

from typing import List, Optional

import torch
from torch import nn


class Bottleneck(nn.Module):
    """
    Bottleneck layer.

    :param in_dim: Number input features.
    :param out_dim: Number of output features. Default is just a score.
    :param hidden_sizes: List with hidden layer sizes.
    :param activations: Name of the activation function to be used in the hidden layers.
    :param final_activation: Name of the final activation function if any.
    :param dropout: dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: List[int] = [3072, 256],
        activations: str = "Sigmoid",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(nn.Dropout(dropout))
        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation):
            return getattr(nn, activation)()

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)