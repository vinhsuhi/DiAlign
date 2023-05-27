import torch 
import torch.nn.functional as F
from src.diffusion.diffusion_utils import kl_divergence_with_probs

def loss_ce(S, y, reduction='mean', t=None):
    '''
    Cross entropy loss
    '''
    EPS = 1e-8
    assert reduction in ['none', 'mean', 'sum']
    val = S[y.bool()]
    nll = -torch.log(val + EPS)
    if t is not None:
        nll = nll * t
    return nll if reduction == 'none' else getattr(torch, reduction)(nll)

def loss_lvb_t(X0, p_probX0, noisy_data, update_info):

        '''
        Returns the KL for one term in the ELBO (time t) (loss L_t)

        This assumes X0 is a sample from from x_0, from which we draw samples from 
        q(x_t | x_0) and then compute q(x_{t-1} | x_t, x_0) following the LaTeX. This 
        is the KL divergence for terms L_1 through L_{T-1}

        args:
            X0: a sample from p(data) (or q(x0))
            p_probX0: a distribution over categories outputed by the denoising model
            noisy_data: contains useful info about Xt
            s_mask
            num_classes: bs x 1 number of target nodes in each batch
            bs: batch_size
            mask_translation
        '''
        Xt = noisy_data['Xt']
        Qtb, Qsb, Qt = noisy_data['Qtb'], noisy_data['Qsb'], noisy_data['Qt']
        
        # q(x_{t-1} | x_t, x_0)
        posterior_true = diffusion_utils.compute_posterior_distribution(X0, Xt, Qt, Qsb, Qtb)
        # p(x_{t-1} | x_t)
        posterior_pred = diffusion_utils.compute_posterior_distribution(p_probX0, Xt, Qt, Qsb, Qtb)

        # Reshape and filter masked rows
        posterior_true, posterior_pred = diffusion_utils.mask_distributions_align(posterior_true, posterior_pred, update_info['s_mask'])

        return kl_divergence_with_probs(posterior_true, posterior_pred).mean()



