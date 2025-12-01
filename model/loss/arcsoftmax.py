import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ArcLoss(nn.Module):
    """
    ArcFace Loss: L = -log( e^(s * cos(theta + m)) / (e^(s * cos(theta + m)) + sum_{j!=y} e^(s * cos(theta_j))) )
    """
    def __init__(self, s=64.0, m=0.5):
        super(ArcLoss, self).__init__()
        self.s = s 
        self.m = m     

    def forward(self, cosine, labels):
        if labels.ndim == 2:
            labels = labels.squeeze(1)
        
        # ensure shapes match
        assert cosine.size(0) == labels.size(0), f"Batch size mismatch: cosine {cosine.size(0)}, labels {labels.size(0)}"
        
        cosine_with_margin = cosine.clone()
        indices = torch.arange(labels.size(0), device=cosine.device)
        cosine_with_margin[indices, labels] = torch.cos(torch.acos(cosine[indices, labels]) + self.m)

        logits = cosine_with_margin * self.s
        loss = F.cross_entropy(logits, labels)
        return loss
