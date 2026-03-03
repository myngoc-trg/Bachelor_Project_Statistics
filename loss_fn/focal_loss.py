from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma = 2.0, reduction = 'mean'):
        """
        Multi-class focal loss
        
        FL(pt) = - alpha_t * (1-pt)^gamma * log(pt)
        
        Args:
            - alpha: tensor of shape [num_classes] or None
                    If provided, acts as class weighting
                    If minority class is severely underrespresented, can set a lower alpha for majority clas and a higher alpha for the minority
                    
            - gamma: focusing parameter (default=2).
                     When gamma = 0, same as cross entropy loss
                     When gamme inccreases, the loss assigned to well-classified (high pt) shrinks, while harder samples (low pt) are given more weight
                     
            - reduction: 'mean','sum', or'none'
        """
    
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Shape:
            logits: [BATCHSIZE, num_classes]
            targets: [BATCHSIZE] (class indices)
        """
        log_probs = F.log_softmax(logits, dim=1) #[BATCHSIZE, num_classes]
        probs = torch.exp(log_probs)
        
        # select probabilities corresponding to the true class
        targets = targets.view(-1,1) # reshape from [BATCHSIZE] to [BATCHSIZE,1]
        
        
        """
        .gather(1, targets) i.e. 
         - For each row b, select elements from row b using the column indices provided in targets[b]
         - Result shape becomes [BATCHSIZE, 1] because we pick 1 value per row
         
         Example:
         - If sample 0 has target 3, pick log_probs[0,3]
         
         Then .squeeze(1) removes the singleton dimension from [BATCHSIZE, 1], giving shape [BATCHSIZE]
        """
        log_pt = log_probs.gather(1, targets).squeeze(1) #[BATCHSIZE]
        pt = probs.gather(1, targets).squeeze(1) #[BATCHSIZE]
        
        # compute focal term
        focal_term = (1 - pt) ** self.gamma

        # apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=0,index=targets.squeeze(1))
            loss = - alpha_t * focal_term * log_pt
        else:
            loss = - focal_term * log_pt
        
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        