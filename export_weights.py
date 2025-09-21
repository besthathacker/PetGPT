import torch
import numpy as np
from transformers import GPT2LMHeadModel

# Load a small GPT model
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Convert weights to numpy
weights = {name: param.detach().numpy() for name, param in model.named_parameters()}
np.savez("tinygpt_weights.npz", **weights)

print("✅ Weights exported successfully!")
