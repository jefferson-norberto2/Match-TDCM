import torch
import torch.nn.functional as F

def soft_focal_loss(pred, gt_score, alpha=2.0, beta=4.0):
    # Clamp predictions to prevent gradient vanishing/exploding
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6) 
    
    # Set values >= 0.5 as positive samples (creates a binarized mask)
    pos_inds = gt_score.ge(0.5).float() 
    # Set values < 0.5 as negative samples
    neg_inds  = gt_score.lt(0.5).float() 
    
    # Background weights
    neg_weights = torch.pow(1 - gt_score, beta) 
    loss = 0
    
    # Positive sample loss
    pos_loss = torch.log(pred) * torch.pow(gt_score - pred, alpha) * pos_inds
    # Negative sample loss
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    # Number of positive samples
    num_pos = pos_inds.float().sum() 
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def center_loss(pred, gt):
    # Clamp predictions to prevent gradient vanishing/exploding
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6) 
    
    # Set values >= 0.5 as positive samples (creates a binarized mask)
    pos_mask = gt.ge(0.5).float() 
    # Set values < 0.5 as negative samples
    neg_mask  = gt.lt(0.5).float() 
    
    # Positive sample loss
    # pos_loss = F.binary_cross_entropy(pred, gt, reduction="none") * pos_mask
    pos_loss = (gt * torch.log(pred) + (1-gt) * torch.log(1-pred)) * pos_mask
    
    # Negative sample loss --- mask(1-gt) are all 1s
    neg_loss = torch.log(1 - pred) * neg_mask
    
    # Number of positive samples (clamped to at least 1 to avoid division by zero)
    num_pos = pos_mask.sum().clamp(min=1.0) 
    
    loss = - (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss

def reg_loss(pred, gt_param, mask):
    regr_loss = F.smooth_l1_loss(pred * mask, gt_param * mask, reduction='sum')
    regr_loss = regr_loss / (mask.sum() + 1e-6)

    return regr_loss 

def Sign_Loss(pred, gt_param, mask):
    # Accuracy calculation
    acc = ((pred.round() == gt_param)* mask).sum() / (mask.sum() + 1e-6)  
    
    pos_weight = (gt_param * mask).sum() / (mask.sum() + 1e-6)
    neg_weight = 1 - pos_weight
    
    # reduction='none': Keep the original shape without averaging or summing for subsequent weighting.
    sign_loss = F.binary_cross_entropy_with_logits(pred * mask, gt_param * mask, reduction='none') 
    
    weighted_loss = sign_loss * (gt_param * pos_weight + (1-gt_param) * neg_weight)
    sign_loss = (weighted_loss * mask).sum() / (mask.sum() + 1e-6) 

    # The purpose of `Sign_Loss` is to measure how well the prediction `pred` matches 
    # the ground truth `gt_param` in terms of sign (positive/negative).
    # The final `loss` reflects the model's error in sign prediction.
    return sign_loss, acc