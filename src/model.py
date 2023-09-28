from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . configuration import GPT2Configuration


class GPT2Embedder(nn.Module):

  def __init__(self, configuration: GPT2Configuration) -> None:
    super().__init__()

    self.embedding0 = nn.Embedding(num_embeddings=configuration.vocabulary_size, embedding_dim=configuration.decoder_dimension)
    self.embedding1 = nn.Embedding(num_embeddings=configuration.vocabulary_size, embedding_dim=configuration.decoder_dimension)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:

    positions = torch.arange(0, len(x))
    x = self.embedding0(x) + self.embedding1(positions)

    return x
    

class GPT2Attention(nn.Module):

  def __init__(self, configuration: GPT2Configuration) -> None:
    super().__init__()

    self.linear0 = nn.Linear(in_features=configuration.decoder_dimension, out_features=configuration.head_dimension, bias=False)  # WQ.
    self.linear1 = nn.Linear(in_features=configuration.decoder_dimension, out_features=configuration.head_dimension, bias=False)  # WK.
    self.linear2 = nn.Linear(in_features=configuration.decoder_dimension, out_features=configuration.head_dimension, bias=False)  # WV.
  
  def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    score = q @ k.transpose(-2, -1)
    score = score / math.sqrt(k.shape[-1])
    score = score.masked_fill(mask == 0, -torch.inf)
    score = score.softmax(dim=-1)
    score = score @ v

    return score

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    q = self.linear0(x)
    k = self.linear1(x)
    v = self.linear2(x)

    return self.attention(q, k, v, mask)  # Use F.scaled_dot_product_attention for optimization.


class GPT2MultiHeadAttention(nn.Module):

  def __init__(self, configuration: GPT2Configuration) -> None:
    super().__init__()

    self.heads = nn.ModuleList([GPT2Attention(configuration) for _ in range(configuration.number_of_heads)])
    self.linear0 = nn.Linear(in_features=configuration.head_dimension * configuration.number_of_heads, out_features=configuration.decoder_dimension, bias=False)  # W0.

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
    x = self.linear0(x)

    return x


class GPT2MLP(nn.Module):

  def __init__(self, configuration: GPT2Configuration) -> None:
    super().__init__()
    
    self.linear0 = nn.Linear(configuration.decoder_dimension, configuration.decoder_dimension * 4)
    self.linear1 = nn.Linear(configuration.decoder_dimension * 4, configuration.decoder_dimension)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    
    x = self.linear0(x)
    x = F.relu(x)
    x = self.linear1(x)
    
    return x


class GPT2Decoder(nn.Module):

  def __init__(self, configuration: GPT2Configuration) -> None:
    super().__init__()

    self.multi_head_attention = GPT2MultiHeadAttention(configuration)
    self.mlp = GPT2MLP(configuration)
    self.layer_normalization_0 = nn.LayerNorm(normalized_shape=configuration.decoder_dimension)
    self.layer_normalization_1 = nn.LayerNorm(normalized_shape=configuration.decoder_dimension)
  
  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    x = x + self.multi_head_attention(x, mask)
    x = self.layer_normalization_0(x)
    x = x + self.mlp(x)
    x = self.layer_normalization_1(x)

    return x
