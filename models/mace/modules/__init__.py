# The code is taken from geometric-dojo repository (https://github.com/chaitjo/geometric-gnn-dojo/tree/main) and reduced
# to the minimum number of the relevant parts for the current application.


###########################################################################################
# This directory contains an implementation of MACE, with minor adaptations
#
# Paper: MACE: Higher Order Equivariant Message Passing Neural Networks
#        for Fast and Accurate Force Fields, Batatia et al.
#
# Original repository: https://github.com/ACEsuit/mace
###########################################################################################

from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    EquivariantProductBasisBlock,
    RadialEmbeddingBlock,
)

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "RadialEmbeddingBlock",
    "EquivariantProductBasisBlock",
]
