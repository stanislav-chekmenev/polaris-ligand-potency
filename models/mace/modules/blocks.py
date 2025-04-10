###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Union

import torch.nn.functional
from e3nn import nn, o3

from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class EquivariantProductBasisBlock(torch.nn.Module):
    """
    Equivariant block for building O(3) equivariant higher order message passing neural network.

    Args:
    - node_feats_irreps (o3.Irreps): Node features irreps
    - target_irreps (o3.Irreps): Target output irreps
    - correlation (Union[int, Dict[str, int]]): Correlation order or body order - 1
    - element_dependent (bool): Whether the block is element dependent
    - use_sc (bool): Whether to use scipped connections
    - batch_norm (bool): Whether to use batch normalization
    - num_elements (Optional[int]): Number of elements for symmetric contraction.
    """

    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        element_dependent: bool = True,
        use_sc: bool = True,
        batch_norm: bool = False,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            element_dependent=element_dependent,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.batch_norm = nn.BatchNorm(target_irreps) if batch_norm else None

    def forward(
        self, node_feats: torch.Tensor, sc: Optional[torch.Tensor], node_attrs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(node_feats)
        if self.batch_norm:
            out = self.batch_norm(out)
        if self.use_sc:
            out = out + sc
        return out


# nonlinearities = {1: torch.nn.SiLU(), -1: torch.nn.Tanh()}
