from dataclasses import dataclass
from typing import *
import torch


@dataclass
class Transformer1DModelOutput:
    sample: torch.FloatTensor
    aux_loss: torch.FloatTensor
