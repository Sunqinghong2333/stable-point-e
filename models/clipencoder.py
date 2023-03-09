import torch
from torch.nn import Module

import clip


class CLIPEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip_model, self.preprocess = clip.load(
            args.clip_name, device=args.device, download_root=args.clip_root
        )
        self.clip_model.to(torch.float32)
        self._tokenize = clip.tokenize


    def encode(self, texts):

        """
        Embed text prompts as an [N x D] tensor.
        """
        # if embeddings:
        #     return embeddings
        # else:

        enc = self.clip_model.encode_text(
            self._tokenize(list(texts), truncate=True).to(self.args.device)
        ).float()
        code = enc / torch.linalg.norm(enc, dim=-1, keepdim=True)
        return code

