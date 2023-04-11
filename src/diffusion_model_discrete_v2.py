import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from tqdm import tqdm

from models.transformer_model import GraphTransformerMatching
from diffusion.noise_schedule import DiscreteUniformTransitionAlign, PredefinedNoiseScheduleDiscrete
from src.diffusion import diffusion_utils
from src import utils
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC


def generate_y(y_col, device):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, test_data=None):
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name # graph-tf-model
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps # 500
        self.model = GraphTransformerMatching(scalar_dim=cfg.model.scalar_dim, num_layers=cfg.model.num_layers, 
                                              ori_feat_dim=cfg.dataset.ori_dim, embed_dim=cfg.model.embed_dim, cat=cfg.model.cat, lin=cfg.model.lin, dropout=cfg.model.dropout)
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps) # done

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransitionAlign()

        self.test_data = test_data
        self.save_hyperparameters()
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0


    def train_loss(self, S, y, reduction='mean'):
        EPS = 1e-8
        assert reduction in ['none', 'mean', 'sum']
        if not S.is_sparse:
            val = S[y[0], y[1]]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            mask = S.__idx__[y[0]] == y[1].view(-1, 1)
            val = S.__val__[[y[0]]][mask]
        nll = -torch.log(val + EPS)
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)


    def sub_forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t 
        name_s, name_t = data.name_s, data.name_t
        batch_s, batch_t = data.x_s_batch, data.x_t_batch

        x_s_, s_mask = to_dense_batch(x_s, batch_s, fill_value=0)
        x_t_, t_mask = to_dense_batch(x_t, batch_t, fill_value=0)
        batch_size = x_s_.shape[0]

        graph_s_data = {'x': x_s, 'edge_index': edge_index_s, 'edge_attr': edge_attr_s, 'name': name_s, 'batch': batch_s}
        graph_t_data = {'x': x_t, 'edge_index': edge_index_t, 'edge_attr': edge_attr_t, 'name': name_t, 'batch': batch_t}

        mask_align = s_mask.unsqueeze(2) * t_mask.unsqueeze(1)
        mask_transition = t_mask.unsqueeze(2) * t_mask.unsqueeze(1)

        align_matrix = torch.zeros((batch_size, s_mask.shape[1], t_mask.shape[1])).to(x_s.device)
        
        batch_num_nodes = list()
        for j in range(batch_size):
            this_y = generate_y(data[j].y, device=x_s.device)
            align_matrix[j][this_y[0], this_y[1]] = 1.
            batch_num_nodes.append(len(data[j].x_t))

        return x_s, x_t, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data


    def training_step(self, data, i):
        # self.eval()
        # self.validation_step(data, i)
        x_s, x_t, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data = self.sub_forward(data)
        noisy_data = self.apply_noise(align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, x_s.device)
        pred = self.forward(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)
        loss = self.train_loss(pred, generate_y(data.y, device=x_s.device))
        print(loss)
        return {'loss': loss}


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())


    def to_test(self, data_cate):
        correct = num_examples = 0

        loader = DataLoader(data_cate, self.cfg.train.batch_size, shuffle=False, follow_batch=['x_s', 'x_t'])
        correct = num_examples = 0
        while (num_examples < 1000):
            for dt in loader:
                dt = dt.to(self.device)
                y = generate_y(dt.y, self.device)
                sample, collapsed_sample = self.sample_batch(dt)
                correct += self.acc(collapsed_sample, y, reduction='sum')
                num_examples += y.size(1)

                if num_examples > 1000:
                    return correct / num_examples
    
    def acc(self, S, y, reduction='mean'):
        r"""Computes the accuracy of correspondence predictions.
        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        correct = (pred == y[1]).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct


    def validation_step(self, data, i):
        print('ahihi')
        accs = [100 * self.to_test(data_cate) for data_cate in self.test_data]
        accs += [sum(accs) / len(accs)]
        print(' '.join([c[:5].ljust(5) for c in PascalVOC.categories] + ['mean']))
        print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))
        

    def validation_epoch_end(self, outs) -> None:
        pass

    def on_test_epoch_start(self) -> None:
        print("Starting test...")

    def test_step(self, data, i):
        pass

    def test_epoch_end(self, outs) -> None:
        pass

    def apply_noise(self, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, node_mask, device):
        '''
        mask_transition: 64 x 19 x 19
        mask_align: 64 x 19 x 19
        align_matrix: 64 x 19 x 19
        node_mask: 64 x 19
        '''
        batch_num_nodes = torch.LongTensor(batch_num_nodes).to(device).reshape(-1, 1) # 64 x 1
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(batch_size,), device=device)
        s_int = t_int - 1

        t_float = t_int / self.T 
        s_float = s_int / self.T 

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float) 
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float) # this is okay!

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, batch_num_nodes, batch_size=batch_size, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # Compute transition probabilities
        probX = align_matrix @ Qtb  # (bs, n, dx_out)

        def sample_discrete_features_align(probX, node_mask):
            ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
                :param probX: bs, n, dx_out        node features
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
            return X_t

        sampled_t = sample_discrete_features_align(probX, node_mask)

        align_t = F.one_hot(sampled_t, num_classes=mask_transition.shape[1])

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'noise_align': align_t, 'mask_align': mask_align}
        return noisy_data


    def forward(self, noisy_data, s_mask, t_mask, graph_s_data, graph_t_data):
        return self.model(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)


    @torch.no_grad()
    def sample_batch(self, data, number_chain_steps=50):
        '''
        TODO: save trajectory and visualize
        '''
        _, _, _, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data = self.sub_forward(data)
        lim_dist = mask_align / (mask_align.sum(dim=-1).reshape(batch_size, mask_align.shape[1], 1) + 1e-8)
        lim_dist[~s_mask] = 1 / lim_dist.shape[-1]
        def sample_discrete_feature_noise():
            sample = lim_dist.flatten(end_dim=-2).multinomial(1).reshape(batch_size, -1)
            long_mask = s_mask.long()
            sample = sample.type_as(long_mask)

            sample = F.one_hot(sample, num_classes=lim_dist.shape[-1]).float()
            return sample

        z_T = sample_discrete_feature_noise()
        x = z_T
        for s_int in tqdm(reversed(range(0, self.T))):
            s_array = s_int * torch.ones((batch_size,)).float()
            t_array = s_array + 1
            s_norm = s_array / self.T 
            t_norm = t_array / self.T 

            sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, x, batch_num_nodes, mask_transition, mask_align, s_mask, t_mask, graph_s_data, graph_t_data)
            x = sampled_s

        collapsed_x = x[s_mask]
        return x, collapsed_x


    def sample_p_zs_given_zt(self, s, t, align_data, batch_num_nodes, 
                                mask_transition, mask_align, s_mask, t_mask, graph_s_data, graph_t_data):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = align_data.shape

        beta_t = self.noise_schedule(t_normalized=t) # 
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s) # 64 x 1
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t) # 64 x 1

        batch_num_nodes = torch.LongTensor(batch_num_nodes).to(self.device).reshape(-1, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qt = self.transition_model.get_Qt(beta_t, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        # Neural net predictions
        noisy_data = {'t': t, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'noise_align': align_data, 'mask_align': mask_align} 

        #noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}

        pred_X = self.forward(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)
        pred_X, _ = to_dense_batch(pred_X, graph_s_data['batch'], fill_value=0)

        p_s_and_t_given_0_X =  diffusion_utils.compute_batched_over0_posterior_distribution(X_t=align_data,
                                                                                            Qt=Qt,
                                                                                            Qsb=Qsb,
                                                                                            Qtb=Qtb)

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        def sample_discrete_features(probX, node_mask):
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
            X_t_ = probX.multinomial(1)                                  # (bs * n, 1)
            X_t_ = X_t_.reshape(bs, n)     # (bs, n)

            return X_t_

        sampled_s = sample_discrete_features(prob_X, node_mask=s_mask)

        X_s = F.one_hot(sampled_s, num_classes=mask_align.shape[1]).float()
        return X_s
