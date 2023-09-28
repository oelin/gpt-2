from dataclasses import dataclass


@dataclass
class GPT2Configuration:
  
  vocabulary_size: int = 10_000 
  sequence_length: int = 256
  decoder_dimension: int = 256 
  head_dimension: int = 256
  number_of_decoders: int = 4
  number_of_heads: int = 8
