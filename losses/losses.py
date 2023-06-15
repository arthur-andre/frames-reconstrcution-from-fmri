import torch
import torch.nn.functional as F


def cosine_loss(r, r_hat):
    r_norm = torch.norm(r, dim=1, keepdim=True)  # Compute the norms of tensor r along the batch dimension
    r_hat_norm = torch.norm(r_hat, dim=1, keepdim=True)  # Compute the norms of tensor r_hat along the batch dimension

    zero_mask = (r_norm == 0) | (r_hat_norm == 0)  # Identify indices where either vector is a zero vector
    zero_mask = zero_mask.squeeze()  # Remove the extra dimension

    # Handle zero vectors
    if zero_mask.any():
        loss = torch.zeros_like(r_norm)  # Initialize loss tensor with zeros
        loss[zero_mask] = torch.tensor(0., requires_grad=True)  # Set loss values to 0 where either vector is zero
        return loss

    cosine_angle = torch.sum(r * r_hat, dim=1) / (r_norm * r_hat_norm)  # Compute the cosine of the angles
    loss = 1 - cosine_angle  # Calculate the loss as 1 minus the cosine of the angles

    return loss

def fmri_loss(r, r_hat, alpha):
    """
    Compute the LE loss between the fMRI recording and its prediction
    
    Args:
    - r: tensor of shape (batch_size, num_voxels)
    - r_hat: tensor of shape (batch_size, num_voxels)
    - alpha: float hyper-parameter
    
    Returns:
    - loss: tensor of shape (batch_size,)
    """

    #print("r",r.shape)
    #print("r_hat",r_hat.shape)


    l2_loss = F.mse_loss(r, r_hat, reduction='none').sum(dim=1)
    #print("l2",l2_loss)
    #cosine_sim = F.cosine_similarity(r, r_hat, dim=1)
    #angle = torch.acos(cosine_sim)
    #angle = cosine_loss(r, r_hat)
    #print("ang",angle)

    #loss = l2_loss + alpha * angle
    #print(loss)
    return l2_loss



