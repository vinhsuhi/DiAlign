import torch
from torch.nn import functional as F
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from src.utils import PlaceHolder


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def sample_gaussian(size):
    x = torch.randn(size)
    return x


def sample_gaussian_with_mask(size, node_mask):
    x = torch.randn(size)
    x = x.type_as(node_mask.float())
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_alpha_bar_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def linear_beta_schedule_discrete(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    # scalx = 1000 / timesteps
    steps = timesteps + 1
    beta_start = 0
    beta_end = 1
    return torch.linspace(beta_start, beta_end, steps, dtype = torch.float64).numpy()


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def gaussian_KL(q_mu, q_sigma):
    """Computes the KL distance between a normal distribution and the standard normal.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch((torch.log(1 / q_sigma) + 0.5 * (q_sigma ** 2 + q_mu ** 2) - 0.5))


def cdf_std_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def sigma_and_alpha_t_given_s(gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size):
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_feature_noise(X_size, E_size, y_size, node_mask):
    """Standard normal noise for all features.
        Output size: X.size(), E.size(), y.size() """
    # TODO: How to change this for the multi-gpu case?
    epsX = sample_gaussian(X_size)
    epsE = sample_gaussian(E_size)
    epsy = sample_gaussian(y_size)

    float_mask = node_mask.float()
    epsX = epsX.type_as(float_mask)
    epsE = epsE.type_as(float_mask)
    epsy = epsy.type_as(float_mask)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(epsE)
    indices = torch.triu_indices(row=epsE.size(1), col=epsE.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    epsE = epsE * upper_triangular_mask
    epsE = (epsE + torch.transpose(epsE, 1, 2))

    assert (epsE == torch.transpose(epsE, 1, 2)).all()

    return PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)


def sample_normal(mu_X, mu_E, mu_y, sigma, node_mask):
    """Samples from a Normal distribution."""
    # TODO: change for multi-gpu case
    eps = sample_feature_noise(mu_X.size(), mu_E.size(), mu_y.size(), node_mask).type_as(mu_X)
    X = mu_X + sigma * eps.X
    E = mu_E + sigma.unsqueeze(1) * eps.E
    y = mu_y + sigma.squeeze(1) * eps.y
    return PlaceHolder(X=X, E=E, y=y)


def check_issues_norm_values(gamma, norm_val1, norm_val2, num_stdevs=8):
    """ Check if 1 / norm_value is still larger than 10 * standard deviation. """
    zeros = torch.zeros((1, 1))
    gamma_0 = gamma(zeros)
    sigma_0 = sigma(gamma_0, target_shape=zeros.size()).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sigma_0 * num_stdevs > 1. / max_norm_value:
        raise ValueError(
            f'Value for normalization value {max_norm_value} probably too '
            f'large with sigma_0 {sigma_0:.5f} and '
            f'1 / norm_value = {1. / max_norm_value}')


def sample_discrete_features(probX, probE, node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X or E
    
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T | q(x_{t-1} | x_t, x_0)
    '''
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)    # (bs, N, d)


    return prob


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def mask_distributions_align(true_align, pred_align, s_mask):
    pred_align = pred_align + 1e-7
    pred_align = pred_align / pred_align.sum(dim=-1, keepdim=True)

    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_align = torch.zeros(true_align.size(-1), dtype=torch.float, device=true_align.device) # num_target_nodes * 1
    row_align[0] = 1.

    true_align[~s_mask] = row_align
    pred_align[~s_mask] = row_align

    return true_align, pred_align


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    # Add a small value everywhere to avoid nans
    pred_X = pred_X + 1e-7
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)

    pred_E = pred_E + 1e-7
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)


    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)

######## I WILL TRANSFER SOME FUNCTIONS FROM FILE DIFFUSION MODEL DISCRETE HERE

def kl_divergence_with_probs(p = None, q = None, epsilon = 1e-20):
    """Compute the KL between two categorical distributions from their probabilities.

    Args:
        p: [..., dim] array with probs for the first distribution.
        q: [..., dim] array with probs for the second distribution.
        epsilon: a small float to normalize probabilities with.

    Returns:
        an array of KL divergence terms taken over the last axis.
    """
    log_p = torch.log(p + epsilon)
    log_q = torch.log(q + epsilon)
    kl = torch.sum(p * (log_p - log_q), dim=-1)

    ## KL divergence should be positive, this helps with numerical stability
    loss = F.relu(kl)
    return loss

def sample_discrete_features_align_wor(q_probXt, s_mask, t_mask, device):
    '''
    Sample features from multinomial distribution (w/o replacement) with given probabilities
    '''
    bs, n, _ = q_probXt.shape
    q_probXt[~s_mask] = 1 / q_probXt.shape[-1]

    samples = list()
    for i in range(len(q_probXt)): # iterate over batch samples
        this_probX = q_probXt[i].clone()
        this_s_mask = s_mask[i]
        this_t_mask = t_mask[i]
        num_nodes_src = this_s_mask.sum().item()
        num_nodes_trg = this_t_mask.sum().item()
        random_indices_src = torch.randperm(num_nodes_src).to(device) # keep this
            
        srcs = list()
        trgs = list()
        for j in range(len(random_indices_src)):
            src_node = random_indices_src[j].item()
            src_prob = (this_probX[src_node] + 1e-12) / (this_probX[src_node] + 1e-12).sum()
            trg_node = src_prob.multinomial(1).item()
            this_probX[:, trg_node] = 0
            srcs.append(src_node)
            trgs.append(trg_node)

        srcs = torch.LongTensor(srcs).to(device)
        trgs = torch.LongTensor(trgs).to(device)

        if this_probX.shape[0] > num_nodes_src:
            left_over_src = torch.arange(num_nodes_src, this_probX.shape[0]).to(device)
            left_over_trg = [this_probX.shape[1] - 1] * len(left_over_src)
            left_over_trg = torch.LongTensor(left_over_trg).to(device)
            srcs = torch.cat((srcs, left_over_src))
            trgs = torch.cat((trgs, left_over_trg))
        
        samples.append(trgs[srcs.sort()[1]])
    Xt = torch.stack(samples) # (bs, n)
    return Xt


def sample_discrete_features_align_wr(q_probXt, s_mask):
    ''' Sample features from multinomial distribution with given probabilities q_probXt
        :param q_probXt: bs, n, dx_out        node features
    '''
    bs, n, _ = q_probXt.shape
    # Noise X
    # The masked rows should define probability distributions as well
    q_probXt[~s_mask] = 1 / q_probXt.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    q_probXt = q_probXt.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    Xt = q_probXt.multinomial(1)                                  # (bs * n, 1)
    Xt = Xt.reshape(bs, n)     # (bs, n)
    return Xt


def metropolis_hastings_one_batch(prob, n_episodes):
    N = prob.shape[1] # N is the number of target nodes (which is greater or equal number of source nodes)
    n = prob.shape[0] # n is the number of source nodes

    # adding pseudo nodes to graph (so that we have equal N and n), and pseudo alignment prob!
    padded_prob = torch.zeros(N, N).cuda()
    padded_prob[:n,:] = prob
    padded_prob[n:, :] += 1 / N # uniform transition probs for pseudo nodes

    # initialize sigma - a random asignment of source nodes to target nodes, but valid?
    sigma = torch.randperm(N).cuda()
    # initialize pi
    pi = torch.zeros(N).long().cuda()
    # initialize mapping
    # source node `i` is mapped to target node `mapping[i]`
    mapping = torch.argsort(sigma) # oh shit!

    for _ in range(n_episodes): # until converge
        this_prob = padded_prob.clone()
        # step 1
        for i in range(N):
            this_prob_row = this_prob[sigma[i].item()]
            # re-normalize the prob:
            this_prob_row = (this_prob_row + 1e-12) / (this_prob_row + 1e-12).sum()
            # take a sample for the src node
            pi[i] = this_prob_row.multinomial(1).item() # pi[i] = j; sigma[i] -- p
            this_prob[:, pi[i]] = 0 # mask the sampled one
        
        this_prob = padded_prob.clone() # one instance from the batch
        # step 2
        log_prob_pi = torch.log(this_prob[sigma, pi]).sum()
        log_prob_sigma = torch.log(this_prob[sigma, torch.arange(N).cuda()]).sum()

        # log(q(pi|sigma)) = sum()
        log_q_pi_given_sigma = torch.tensor(0.0).cuda()
        this_prob = padded_prob.clone()
        for i in range(N):
            # re-normalize the remaining probs
            this_prob = (this_prob + 1e-12) / (this_prob + 1e-12).sum(dim=1, keepdim=True)
            # take the probability of pair (sigma(i), pi(i))
            # shouldn't it be this_prob[[pi[sigma]]| sigma]
            log_q_pi_given_sigma += torch.log(this_prob[sigma[i], pi[i]])
            this_prob[:, pi[:i]] = 0
        
        log_q_sigma_given_pi = torch.tensor(0.0).cuda()
        this_prob = padded_prob.clone()
        for i in range(N):
            # re-normalize the remaining probs
            this_prob = (this_prob + 1e-12) / (this_prob + 1e-12).sum(dim=1, keepdim=True)
            # take the probability of pair (pi(i), sigma(i))
            log_q_sigma_given_pi += torch.log(this_prob[pi[i], sigma[i]])
            this_prob[:, sigma[:i]] = 0
        
        # if Uniform(0,1) < p(π)·q(σ|π) / (p(σ)·q(π|σ))
        acc_prob = torch.exp(log_prob_pi + log_q_sigma_given_pi - (log_prob_sigma + log_q_pi_given_sigma))
        if np.random.rand() < acc_prob:
            # accept new sigma
            sigma = pi[sigma]
            mapping = torch.argsort(sigma)

        # source node `i` is mapped to target node `mapping[i]`
    return mapping[:n].detach().cpu().numpy()

def sample_by_hungarian(q_probXt, s_mask, t_mask, device):
    bs, n, _ = q_probXt.shape 
    q_probXt[~s_mask] = 0
    samples = list() 
    for b in range(bs):
        this_probXt = q_probXt[b].detach().cpu().numpy() # cost matrix
        srcs, trgs = linear_sum_assignment(1 - this_probXt)
        samples.append(torch.LongTensor(trgs[srcs.sort()]).to(device))
    Xt = torch.cat(samples, dim=0)
    return Xt 


def perturb_sampling_one_batch(prob, this_temperature, device):
    '''
    prob: the posterior of size (n_src_nodes x n_trg_nodes)
    '''
    from torch.distributions.gumbel import Gumbel
    log_prob = torch.log(prob)
    # generate gumbel noise Gumbel(0, 1)
    gumbel_distribution = Gumbel(loc=torch.tensor([0.0]), scale=torch.tensor([this_temperature]))
    # sample N x N gumbel noise instances
    gumbel_noises = gumbel_distribution.sample(log_prob.shape).to(self.device).squeeze()

    # log(p) + e
    combined_ = log_prob + gumbel_noises

    # cost_matrix = - simi_matrix (or alignment matrix)
    cost = -combined_
    cost = cost.detach().cpu().numpy()

    # run Hungarian algorithm
    srcs, trgs = linear_sum_assignment(cost)

    this_sample = trgs[srcs.sort()]
    
    return this_sample
 
 
def sample_discrete_feature_noise_wor(bs, mask_align, s_mask, t_mask, lim_dist, device):
    sample = list()
    for i in range(bs):
        mask_align_i = mask_align[i]
        this_s_mask = s_mask[i]
        this_t_mask = t_mask[i]
        num_nodes_src = this_s_mask.sum().item()
        num_nodes_trg = this_t_mask.sum().item()
        random_indices_src = torch.randperm(num_nodes_src).to(device)
        random_indices_trg = torch.randperm(num_nodes_trg).to(device)
        trg_assigned_to_src = random_indices_trg[:num_nodes_src]

        if (mask_align_i.shape[0] > num_nodes_src):
            left_over_trg = random_indices_trg[num_nodes_src:]
            if mask_align_i.shape[1] > num_nodes_trg:
                left_over_trg = torch.cat((left_over_trg, torch.arange(num_nodes_trg, mask_align_i.shape[1]).to(device)))
            left_over_src = torch.arange(num_nodes_src, mask_align_i.shape[0]).to(device)
            left_over_trg_assigned_to_src = left_over_trg[:len(left_over_src)]
            final_trg_assigned = torch.cat((trg_assigned_to_src, left_over_trg_assigned_to_src))
            final_indices_src = torch.cat((random_indices_src, left_over_src))
        else:
            final_trg_assigned = trg_assigned_to_src
            final_indices_src = random_indices_src

        sample.append(final_trg_assigned[final_indices_src.sort()[1]])
    sample = torch.stack(sample)
    long_mask = s_mask.long()
    sample = sample.type_as(long_mask)

    sample = F.one_hot(sample, num_classes=lim_dist.shape[-1]).float()
    return sample

def gen_inter_info(data, device):
    '''
    speed optimized! DONE!
    '''
    bs = len(data['gt_perm_mat'][0])
    num_classes = data['gt_perm_mat'][0].shape[-1]
    num_variables = data['gt_perm_mat'][0].shape[-2]
    
    src_graph, trg_graph = data['edges'][0], data['edges'][1]
    num_src_nodes = torch.bincount(src_graph.batch)
    num_trg_nodes = torch.bincount(trg_graph.batch)
    
    s_mask = torch.zeros(bs, num_variables).to(device).bool()
    t_mask = torch.zeros(bs, num_classes).to(device).bool()
    
    for idx in range(bs):
        s_mask[idx][:num_src_nodes[idx].item()] = True
        t_mask[idx][:num_trg_nodes[idx].item()] = True
        
    mask_align = s_mask.unsqueeze(2) * t_mask.unsqueeze(1)
    mask_transition = t_mask.unsqueeze(2) * t_mask.unsqueeze(1)
    
    update_info = dict()
    update_info['s_mask'] = s_mask
    update_info['t_mask'] = t_mask
    update_info['mask_align'] = mask_align
    update_info['mask_transition'] = mask_transition
    update_info['bs'] = bs
    update_info['src_batch'] = src_graph.batch 
    update_info['trg_batch'] = trg_graph.batch
    update_info['num_classes'] = t_mask.sum(dim=1, keepdim=True)
    return update_info

def sample_discrete_feature_noise_wr(bs, lim_dist):
    sample = lim_dist.flatten(end_dim=-2).multinomial(1).reshape(bs, -1)
    sample = sample.long()
    sample = F.one_hot(sample, num_classes=lim_dist.shape[-1]).float()
    return sample

# def visualize_batch(self, batch, dtn='train'):
    #     _, _, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, _, _ = self.sub_forward(batch)
    #     poses, gts, edge_srcs, edge_trgs = self.forward_diffusion(align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, batch, dtn=dtn)
    #     self.reverse_diffusion(batch, poses, gts, edge_srcs, edge_trgs, s_mask, t_mask, dtn=dtn)

    
    # def visualise_instance(self, edge_src, edge_trg, gt, name, pos=None):
    #     graph = nx.Graph()
    #     graph.add_edges_from(edge_src.t().detach().cpu().numpy().tolist())
    #     init_index_target = edge_src.max() + 1
    #     target_edges = edge_trg + init_index_target
    #     graph.add_edges_from(target_edges.t().detach().cpu().numpy().tolist())
    #     gt_ = gt.clone()
    #     gt_[1] += init_index_target
    #     gt_ = gt_.t().detach().cpu().numpy().tolist()
    #     graph.add_edges_from(gt_)

    #     gt_dict = dict()
    #     for i in range(len(gt_)):
    #         gt_dict[gt_[i][0]] = gt_[i][1]

    #     if pos is None:
    #         pos = nx.spring_layout(graph)
    #         for k, v in pos.items():
    #             if k < init_index_target:
    #                 try:
    #                     pos[k][0] = pos[gt_dict[k]][0] - 2
    #                     pos[k][1] = pos[gt_dict[k]][1] + 0.05
    #                 except:
    #                     pos[k][0] = pos[k][0] - 2

    #     plt.figure(figsize=(4, 4))
    #     options = {"edgecolors": "tab:gray", "node_size": 80, "alpha": 0.9}
    #     nx.draw_networkx_nodes(graph, pos, nodelist=list(range(init_index_target)), node_color="tab:red", **options)
    #     nx.draw_networkx_nodes(graph, pos, nodelist=list(range(init_index_target, len(pos))), node_color="tab:blue", **options)
    #     nx.draw_networkx_edges(graph, pos)
    #     nx.draw_networkx_edges(graph, pos, edgelist=gt_, width=3, alpha=0.5, edge_color="tab:green")
    #     plt.tight_layout()
    #     plt.savefig(name)
    #     plt.close()
    #     return pos
    

    # def reverse_diffusion(self, batch, poses, gts, edge_srcs, edge_trgs, s_mask, t_mask, dtn='train'):
    #     device = self.device
    #     trajectory = self.sample_batch(batch, return_traj=True)
    #     for s_int in reversed(range(0, self.T + 1)):
    #         align = trajectory[s_int]
    #         for i in range(len(align)): # number of data examples to be visualised
    #             this_align = generate_y(align[i][s_mask[i]].argmax(dim=1), device)
    #             if ((self.T - s_int) % self.step_visual == 0) or (s_int == 0):
    #                 path_visual = 'visuals/{}/sp{}/X_pred_{}.png'.format(dtn, i, self.T - s_int)
    #                 self.visualise_instance(edge_srcs[i], edge_trgs[i], this_align, path_visual, poses[i])


    # def forward_diffusion(self, X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, batch, dtn='train'):
    #     device = self.device
    #     gts = list()
    #     edge_srcs = list()
    #     edge_trgs = list()
    #     poses = list()
    #     for i in range(bs):
    #         this_batch = batch[i]
    #         edge_srcs.append(this_batch['edge_index_s'])
    #         edge_trgs.append(this_batch['edge_index_t'])
    #         gts.append(generate_y(this_batch.y, device))
    #         path_visual = 'visuals/{}/sp{}/'.format(dtn, i)
    #         if not os.path.exists(path_visual):
    #             os.makedirs(path_visual)
    #         poses.append(self.visualise_instance(edge_srcs[-1], edge_trgs[-1], gts[-1], '{}/X_0.png'.format(path_visual)))

    #     for t_int in range(0, self.T):
    #         noisy_sample = self.apply_noise(X0, mask_align, mask_transition, bs, s_mask, t_mask, t_int=t_int)
    #         Xt = noisy_sample['Xt']
    #         for i in range(bs):
    #             this_noisy_alignment = generate_y(Xt[i][s_mask[i]].argmax(dim=1), device)
    #             if ((t_int + 1) % self.step_visual == 0) or (t_int + 1) == self.T:
    #                 path_visual = 'visuals/{}/sp{}/X_{}.png'.format(dtn, i, t_int + 1)
    #                 self.visualise_instance(edge_srcs[i], edge_trgs[i], this_noisy_alignment, path_visual, poses[i])
    #     return poses, gts, edge_srcs, edge_trgs

