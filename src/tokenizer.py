import os
from pathlib import Path

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class GPT2Tokenizer:

  def __init__(self, model_path: Path):
    self.processor = SentencePieceProcessor(model_file=model_path + '.model')
    self.bos_id = self.processor.bos_id()
    self.eos_id = self.processor.bos_id()
    self.pad_id = self.processor.pad_id()

  @classmethod
  def train(
      self,
      configuration: GPT2Configuration,
      input_path: Path,
      model_path: Path,
  ) -> None:

    SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_path,
        vocab_size=configuration.vocabulary_size,
    )

    return GPT2Tokenizer(model_path)

  @property
  def vocabulary_size(self) -> int:
    return self.processor.vocab_size()

  def encode(
      self,
      input: str,
      use_bos: bool = True,
      use_eos: bool = False,
      use_pad: bool = False,
  ) -> torch.Tensor:

    tokens = self.processor.encode(input)

    if use_bos: tokens = [self.bos_id] + tokens
    if use_eos: tokens = tokens + [self.eos_id]
    #if use_pad: tokens = tokens + [eslf.pad_id] * (maximum_length - len(tokens))

    return torch.tensor(tokens, dtype=torch.int)

  def decode(self, tokens: torch.Tensor) -> str:
    return self.processor.decode(tokens.tolist())
