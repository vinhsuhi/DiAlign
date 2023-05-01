import numpy as np
import torch
import math
import sys

sys.path.append("./")

from src import utils
from src.diffusion import diffusion_utils


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = diffusion_utils.cosine_alpha_bar_schedule(timesteps)
        elif noise_schedule == 'custom':
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2     # (timesteps + 1, )

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]



class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)


    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar[t_int.long().to(self.alphas_bar.device)]


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qtb(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)



class DiscreteUniformTransitionAlign:
    def __init__(self):
        pass

    def get_Qt(self, beta_t, num_classes, batch_size, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).

        num_classes: = num_classes 
        """
        # reshape beta_t
        beta_t_ = beta_t.unsqueeze(1).unsqueeze(-1)
        beta_t_ = beta_t_.to(device)

        # compute 11T / K
        u_x = torch.ones(batch_size, num_classes.max().item(), num_classes.max().item())
        u_x = u_x.to(device)
        u_x = u_x / num_classes.unsqueeze(-1)

        # define eye matrix
        eye_matrix = torch.eye(num_classes.max().item(), device=device).repeat(batch_size, 1).reshape(batch_size, num_classes.max().item(), -1)

        # compute Qt
        q_x = (1 - beta_t_) * eye_matrix + beta_t_ * u_x
        return q_x


    def get_Qtb(self, alpha_bar_t, num_classes, batch_size, device):
        """ Returns t-step transition matrices for align_data, from step 0 to step t.
        Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: q_x (bs, dx, dx).
        """
        # reshape alpha_bar_t
        alpha_bar_t_ = alpha_bar_t.unsqueeze(1).unsqueeze(-1)
        alpha_bar_t_ = alpha_bar_t_.to(device)

        # compute 11T / K
        u_x = torch.ones(batch_size, num_classes.max().item(), num_classes.max().item()) # (bs, M , M) 
        u_x = u_x.to(device)
        u_x = u_x / num_classes.unsqueeze(-1)
        
        # define eye matrix
        eye_matrix = torch.eye(num_classes.max().item(), device=device).repeat(batch_size, 1).reshape(batch_size, num_classes.max().item(), -1)

        # compute Qt_bar
        q_x = alpha_bar_t_ * eye_matrix + (1 - alpha_bar_t_) * u_x
        return q_x


class MarginalUniMatchingTransition:
    def __init__(self, max_num_classes):
        self.g_table = torch.zeros([max_num_classes + 1], dtype = torch.float64)
        curr_denom = 0.0
        self.g_table[0] = 0
        self.g_table[1] = math.log(1e-6)
        self.g_table[2] = 0
        for n in range(3, max_num_classes + 1):
            g = self.logsubexp(
                self.log_factorial(n) - curr_denom,
                torch.logaddexp(
                    torch.logsumexp(
                        self.log_binomial_coefficients(n)[:-1] + self.g_table[torch.arange(n-1, 1, -1)], dim = 0),
                    torch.tensor(0)
                ).item()
            )
            if g > 1e300:
                g -= 1e10
                self.g_table -= 1e10
                curr_denom += 1e10
            self.g_table[n] = g

    def get_Qt(self, beta_t, alpha_bar_t, num_classes, batch_size, device):
        '''beta_t = beta_t.cpu()
        alpha_bar_t = alpha_bar_t.cpu()
        num_classes = num_classes.cpu()

        Qt1b = self.get_Qtb(alpha_bar_t - beta_t, num_classes, batch_size, device)
        Qtb = self.get_Qtb(alpha_bar_t, num_classes, batch_size, device)

        Q = (torch.inverse(Qt1b) @ Qtb).to(device)
        Q /= Q.sum(dim = 1, keepdim = True)

        return Q'''

        Q = self.get_Qtb(beta_t.clamp(min = 0.01, max = 0.99), num_classes, batch_size, device)

        return Q

    def get_Qtb(self, alpha_bar_t, num_classes, batch_size, device):

        alpha_bar_t = alpha_bar_t.cpu()
        num_classes = num_classes.cpu()

        num_classes = num_classes.squeeze()

        nmax = num_classes.max().item()

        v1 = torch.log(1-alpha_bar_t) + self.g_lamda(num_classes-1, alpha_bar_t)

        mask = torch.zeros([num_classes.size(0), nmax-1], dtype = torch.float64)
        mask[:,:] = -torch.inf
        for i in range(num_classes.size(0)):
            mask[i,:num_classes[i]-1] = 0.0

        v2 = torch.logsumexp(torch.stack([
            torch.log(alpha_bar_t) + self.log_Ank(num_classes, i) + self.g_lamda(num_classes - i - 1, alpha_bar_t) for i in range(nmax - 1)
        ], dim = 1) + mask, dim = 1)

        diag_v = torch.exp(v1 - torch.logaddexp(v1, v2))
        ndiag_v = torch.exp(v2 - torch.logaddexp(v1, v2)) / (num_classes - 1)

        u_x = torch.ones(batch_size, num_classes.max().item(), num_classes.max().item()) # (bs, M , M) 
        u_x[:,:,:] = ndiag_v.unsqueeze(1).unsqueeze(2)
        for i in range(batch_size):
            u_x[i].diagonal().fill_(diag_v[i])

        u_x /= u_x.sum(dim = 1, keepdim = True)

        return u_x.to(device)

    def g_lamda(self, ns, alpha):
        n = ns.max().item()
        log_binomials = torch.cat([
            torch.tensor([0], dtype = torch.float64),
            self.log_binomial_coefficients(n),
            torch.tensor([0], dtype = torch.float64)
        ], dim = 0)
        mask = torch.zeros([ns.size(0), n+1], dtype = torch.float64)
        mask[:,:] = -torch.inf
        for i in range(ns.size(0)):
            mask[i,:ns[i]+1] = 0.0
        return torch.logsumexp(
            self.g_table[torch.arange(n,-1,-1)].unsqueeze(0) + log_binomials.unsqueeze(0) + self.log_factors(alpha, n, 0, n) + mask, 
            dim = 1
        )

    @staticmethod
    def log_binomial_coefficients(n):
        """
        Computes the logarithm of the binomial coefficient for (n,1) to (n,n-1) in a single vector.

        Parameters:
        n (int): The value of n.

        Returns:
        torch.Tensor: A 1D tensor containing the logarithm of the binomial coefficients.
        """
        k = torch.arange(1, n, dtype=torch.float64)
        log_n = torch.log(torch.tensor(n, dtype=torch.float64))

        # Compute the logarithm of the binomial coefficient using the log gamma function
        log_bc = math.lgamma(n+1) - torch.lgamma(k+1) - torch.lgamma(n-k+1)

        return log_bc

    @staticmethod
    def log_factors(lamda, n, i_start, i_end):
        lamda = lamda.reshape(-1, 1)
        i = torch.arange(i_start, i_end + 1).unsqueeze(0)
        return torch.log(1-lamda) * i + torch.log(lamda) * (n-i)

    @staticmethod
    def log_factorial(n):
        if n > 1000:
            # Stirling's approximation
            return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
        elif n < 1:
            return -100.0
        else:
            return torch.log(torch.arange(1, n+1)).sum().item()

    def log_Ank(self, ns, k):
        return torch.tensor([self.log_factorial(n) - self.log_factorial(n-k) for n in ns])

    @staticmethod
    def logsubexp(a, b):
        """
        Computes the logarithm of the difference of two exponentiated values.

        Parameters:
        a (float): The logarithm of the first value.
        b (float): The logarithm of the second value.

        Returns:
        float: The logarithm of the difference of the exponentiated values (exp(a) - exp(b)).
        """
        if a == b:
            return float('-inf')
        elif a > b:
            return a + math.log1p(-math.exp(b-a))
        else:
            return b + math.log1p(-math.exp(a-b))


class MarginalUniformTransition:
    def __init__(self, x_marginals, e_marginals, y_classes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qtb(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class AbsorbingStateTransition:
    def __init__(self, abs_state: int, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

        self.u_x = torch.zeros(1, self.X_classes, self.X_classes)
        self.u_x[:, :, abs_state] = 1

        self.u_e = torch.zeros(1, self.E_classes, self.E_classes)
        self.u_e[:, :, abs_state] = 1

        self.u_y = torch.zeros(1, self.y_classes, self.y_classes)
        self.u_e[:, :, abs_state] = 1

    def get_Qt(self, beta_t):
        """ Returns two transition matrix for X and E"""
        beta_t = beta_t.unsqueeze(1)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes).unsqueeze(0)
        return q_x, q_e, q_y

    def get_Qtb(self, alpha_bar_t):
        """ beta_t: (bs)
        Returns transition matrices for X and E"""

        alpha_bar_t = alpha_bar_t.unsqueeze(1)

        q_x = alpha_bar_t * torch.eye(self.X_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return q_x, q_e, q_y


if __name__ == "__main__":

    nt = MarginalUniMatchingTransition(100)

    alpha_bar_t = torch.tensor([0.1, 0.3, 0.8])
    beta_t = torch.tensor([0.01, 0.01, 0.01])
    num_classes = torch.tensor([[6],[9],[20]])

    print(nt.get_Qtb(alpha_bar_t, num_classes, 3, torch.device("cpu")))
    print(nt.get_Qt(beta_t, alpha_bar_t, num_classes, 3, torch.device("cpu")))