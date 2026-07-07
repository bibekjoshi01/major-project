"""
Loading helpers for the Spectra-AASIST3 model.

Two ways to get a model:

1. --hf_repo lab260/Spectra-AASIST3
      Loads straight from the Hugging Face hub with `.from_pretrained`.

2. --model_path /path/to/checkpoint.pth
      Instantiates SpectraAASIST3() locally and loads a state_dict from disk.
"""

import torch


def load_model(device, model_path=None, hf_repo=None):
    """
    Returns a SpectraAASIST3 model in eval() mode on the given device.

    Exactly one of model_path / hf_repo should be provided.
    """
    if not model_path and not hf_repo:
        raise ValueError("Provide either --model_path or --hf_repo.")

    from model import SpectraAASIST3  # noqa: local import, needs model.py on path

    if hf_repo:
        print(f"Loading Spectra-AASIST3 from HF hub: {hf_repo}")
        model = SpectraAASIST3.from_pretrained(hf_repo)
    else:
        print(f"Loading Spectra-AASIST3 from local checkpoint: {model_path}")
        model = SpectraAASIST3()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model
