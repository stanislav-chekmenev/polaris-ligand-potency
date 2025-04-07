import torch

from torch.nn.functional import softmax

import config as cfg


class SelfAttention(torch.nn.Module):
    """
    SelfAttention class.
    This class is used to implement a self-attention mechanism.
    The input features are the atom features, the molecular features, 3d mace features and the barycenters.
    The output features are the attention features.
    """
    def __init__(self, input_dim=cfg.IN_ATTENTION_DIM, num_heads=cfg.NUM_HEADS, num_mol_embeddings=4):
        """
        Initialize the SelfAttention class.
        Parameters:
            - input_dim (int): The input dimension of the model.
            - num_heads (int): The number of attention heads.
            - num_mol_embeddings (int): The number of molecular embeddings.
        """
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.in_proj = torch.nn.Linear(input_dim, input_dim * 3) # Get queries, keys, values
        self.out_proj = torch.nn.Linear(input_dim, input_dim)
        self.num_heads = num_heads
        try:
            input_dim % num_heads == 0
        except:
            raise(ValueError, "Input dimension must be divisible by number of heads")
        self.head_dim = input_dim // num_heads
        self.num_mol_embeddings = num_mol_embeddings


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create an interim shape for processing the data by each head in parallel
        input_shape = x.shape
        batch_size = input_shape[0]
        interim_shape = (batch_size, self.num_mol_embeddings, self.num_heads, self.head_dim)

        # Get Q, K, V
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_size, Num_mol_emb, Dim) -> 
        #  -> (Batch_size, n_heads, Num_mol_emb, head_dim) = [batch_size, num_heads, 4, 16]
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, n_heads, Num_mol_emb, head_dim) @ (Batch_size, n_heads, head_dim. Num_mol_emb) ->
        # -> (Batch_size, n_heads, 4, 16) @ [batch_size, num_heads, 16, 4]
        # -> (Batch_size, n_heads, Num_mol_emb, Num_mol_emb)
        weight = q @ k.transpose(-1, -2) # dim = [Batch_size, n_heads, 4, 4]

        # Scale the weights to avoid explosion of variance
        weight = weight / (self.head_dim ** 0.5)

        # Apply softmax to get the attention weights
        weight = softmax(weight, dim=-1)

        # (Batch_size, n_heads, Num_mol_emb, Num_mol_emb) @ (Batch_size, n_heads, Num_mol_emb, head_dim) ->
        # -> (Batch_size, n_heads, Num_mol_emb, head_dim) = [batch_size, num_heads, 4, 16]
        # Compute weighted average of the values
        output = weight @ v

        # (Batch_size, n_heads, Num_mol_emb, head_dim) -> (Batch_size, Num_mol_emb, n_heads, head_dim)
        output = output.transpose(1, 2)

        # Reshape the output to the original shape -> (Batch_size, Num_mol_emb, input_dim)
        output = output.reshape(input_shape)

        # Apply the output projection
        output = self.out_proj(output)

        return output
