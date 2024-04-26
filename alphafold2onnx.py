"""
Example export script to generate a `.onnx` alphafold2 model.

Uses [alphafold2_pytorch](https://github.com/lucidrains/alphafold2) for a torch implementation of alphafold2.

One tweak was made to the source code in `alphafold2_pytorch/alphafold2.py`. The torch implementation returns
a class for the outputs. This was changed to a tuple or scalar based on the parameterization of the model.

Next Steps: 

    - Translate the original DeepMind Jax implementation to ONNX
    - Use pass real inputs and weights and validate the output results
"""

import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim=256, depth=2, heads=8, dim_head=64, predict_angles=True  # set this to True
)

seq = torch.randint(0, 21, (1, 128))
msa = torch.randint(0, 21, (1, 5, 128))
mask = torch.ones_like(seq).bool()
msa_mask = torch.ones_like(msa).bool()

onnx_inputs = {"seq": seq, "msa": msa, "mask": mask, "msa_mask": msa_mask}

torch.onnx.export(model, onnx_inputs, "alpha_fold2.onnx")

# Now we have the file alpha_fold.onnx
