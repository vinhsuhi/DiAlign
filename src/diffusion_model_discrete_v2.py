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
from src.metrics.align_metrics import AlignAcc
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import networkx as nx 
import matplotlib.pyplot as plt

def generate_y(y_col, device):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.general.name # graph-tf-model
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps # 500
        self.model = GraphTransformerMatching(scalar_dim=cfg.model.scalar_dim, num_layers=cfg.model.num_layers, 
                                              ori_feat_dim=cfg.dataset.ori_dim, embed_dim=cfg.model.embed_dim, cat=cfg.model.cat, lin=cfg.model.lin, dropout=cfg.model.dropout, use_time=cfg.model.use_time)
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps) # done

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransitionAlign()

        # log hyperparameters
        self.save_hyperparameters()

        self.train_acc = AlignAcc()
        self.val_acc = AlignAcc()
        self.test_acc = AlignAcc()
        self.log_every_steps = cfg.general.log_every_steps

        self.train_samples_to_visual = None 
        self.test_samples_to_visual = None


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


    def sub_forward(self, batch):
        x_s, x_t = batch.x_s, batch.x_t
        edge_index_s, edge_index_t = batch.edge_index_s, batch.edge_index_t
        edge_attr_s, edge_attr_t = batch.edge_attr_s, batch.edge_attr_t 
        name_s, name_t = batch.name_s, batch.name_t
        batch_s, batch_t = batch.x_s_batch, batch.x_t_batch

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
            this_y = generate_y(batch[j].y, device=x_s.device)
            align_matrix[j][this_y[0], this_y[1]] = 1.
            batch_num_nodes.append(len(batch[j].x_t))

        return x_s, x_t, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data


    def training_step(self, batch, batch_idx):
        x_s, x_t, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data = self.sub_forward(batch)
        noisy_data = self.apply_noise(align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, x_s.device)
        pred = self.forward(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)
        target = generate_y(batch.y, device=x_s.device)
        loss = self.train_loss(pred, target)
        self.log('train/loss', loss, on_epoch=True, batch_size=target.shape[1])
        return {'loss': loss}


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)


    # def on_fit_start(self) -> None:
    #     self.train_iterations = len(self.trainer.datamodule.train_dataloader())

    def validation_step(self, batch, batch_idx, dataloader_idx):
        category = PascalVOC.categories[dataloader_idx]
        target = generate_y(batch.y, self.device)
        node_batch_size = target.shape[1]
        sample, pred = self.sample_batch(batch)
        test_acc = self.val_acc(pred, target)
        self.log("test/acc_epoch/{}".format(category), test_acc, batch_size=node_batch_size)
        return {'test_acc': test_acc, 'bs': node_batch_size}

    def validation_epoch_end(self, outs):
        accs = list()
        for out_ in outs:
            this_sum = 0
            this_acc = 0
            for out in out_:
                acc, bs = out['test_acc'], out['bs']
                this_sum += bs 
                this_acc += acc * bs
            accs.append(this_acc / this_sum)
        acc = sum(accs) / len(accs)
        self.log("test/acc_epoch/mean", acc)
        print('Evaluation finished')

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()
        self.test_acc.reset()
        print("Start evaluating after {} epochs...".format(self.current_epoch))

    def on_test_epoch_start(self) -> None:


        print('Visualizing output...')

        for batch in self.train_samples_to_visual:
            batch = batch.to(self.device)
            self.visualize_batch(batch, dtn='train')
            break

        for batch in self.test_samples_to_visual:
            batch = batch.to(self.device)
            self.visualize_batch(batch, dtn='test')
            break

        print('Done visualization!')
        print("Starting test...")

    def test_step(self, data, batch_idx, dataloader_idx):
        return self.validation_step(data, batch_idx, dataloader_idx)        

    def test_epoch_end(self, outs) -> None:

        print("Testing finished")
        accs = list()
        for out_ in outs:
            this_sum = 0
            this_acc = 0
            for out in out_:
                acc, bs = out['test_acc'], out['bs']
                this_sum += bs 
                this_acc += acc * bs
            accs.append(this_acc / this_sum)
        acc = sum(accs) / len(accs)
        self.log("test/acc_epoch/MEAN", acc)
        print('Evaluation finished')
    
    def visualize_batch(self, batch, dtn='train'):
        """
        TODO: Visualize forward
        TODO: VIsualize reverse
        """
        _, _, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, node_mask, _, _, _ = self.sub_forward(batch)
        poses, gts, edge_srcs, edge_trgs = self.forward_diffusion(align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, node_mask, batch, dtn=dtn)
        self.reverse_diffusion(batch, poses, gts, edge_srcs, edge_trgs, node_mask, dtn=dtn)

    

    def visualise_instance(self, edge_src, edge_trg, gt, name, pos=None):
        graph = nx.Graph()
        graph.add_edges_from(edge_src.t().detach().cpu().numpy().tolist())
        init_index_target = edge_src.max() + 1
        target_edges = edge_trg + init_index_target
        graph.add_edges_from(target_edges.t().detach().cpu().numpy().tolist())
        gt_ = gt.clone()
        gt_[1] += init_index_target
        gt_ = gt_.t().detach().cpu().numpy().tolist()
        graph.add_edges_from(gt_)

        gt_dict = dict()
        for i in range(len(gt_)):
            gt_dict[gt_[i][0]] = gt_[i][1]

        if pos is None:
            pos = nx.spring_layout(graph)
            for k, v in pos.items():
                if k < init_index_target:
                    try:
                        pos[k][0] = pos[gt_dict[k]][0] - 2
                        pos[k][1] = pos[gt_dict[k]][1] + 0.05
                    except:
                        pos[k][0] = pos[k][0] - 2

        plt.figure(figsize=(4, 4))
        options = {"edgecolors": "tab:gray", "node_size": 80, "alpha": 0.9}
        nx.draw_networkx_nodes(graph, pos, nodelist=list(range(init_index_target)), node_color="tab:red", **options)
        nx.draw_networkx_nodes(graph, pos, nodelist=list(range(init_index_target, len(pos))), node_color="tab:blue", **options)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_edges(graph, pos, edgelist=gt_, width=3, alpha=0.5, edge_color="tab:green")
        plt.tight_layout()
        plt.savefig(name)
        plt.close()
        return pos
    

    def reverse_diffusion(self, batch, poses, gts, edge_srcs, edge_trgs, node_mask, dtn='train'):
        device = self.device
        trajectory = self.sample_batch(batch, return_traj=True)
        for s_int in reversed(range(0, self.T + 1)):
            align = trajectory[s_int]
            print(align.shape)
            for i in range(len(align)):
                this_align = generate_y(align[i][node_mask[i]].argmax(dim=1), device)
                path_visual = '/netscratch/duynguyen/Research/vinh/DiAlign/visuals/{}/sp{}/X_pred_{}.png'.format(dtn, i, self.T - s_int)
                self.visualise_instance(edge_srcs[i], edge_trgs[i], this_align, path_visual, poses[i])


    def forward_diffusion(self, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, node_mask, batch, dtn='train'):
        device = self.device
        # draw the first no-noise graph
        gts = list()
        edge_srcs = list()
        edge_trgs = list()
        poses = list()
        for i in range(batch_size):
            this_batch = batch[i]
            edge_srcs.append(this_batch['edge_index_s'])
            edge_trgs.append(this_batch['edge_index_t'])
            gts.append(generate_y(this_batch.y, device))
            path_visual = '/netscratch/duynguyen/Research/vinh/DiAlign/visuals/{}/sp{}/'.format(dtn, i)
            if not os.path.exists(path_visual):
                os.makedirs(path_visual)
            poses.append(self.visualise_instance(edge_srcs[-1], edge_trgs[-1], gts[-1], '{}/X_0.png'.format(path_visual)))

        for t_int in range(0, self.T):
            noisy_sample = self.apply_noise(align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, node_mask, device, t_int=t_int)
            noisy_alignment = noisy_sample['noise_align']
            for i in range(batch_size):
                this_noisy_alignment = generate_y(noisy_alignment[i][node_mask[i]].argmax(dim=1), device)
                path_visual = '/netscratch/duynguyen/Research/vinh/DiAlign/visuals/{}/sp{}/X_{}.png'.format(dtn, i, t_int + 1)
                self.visualise_instance(edge_srcs[i], edge_trgs[i], this_noisy_alignment, path_visual, poses[i])
        return poses, gts, edge_srcs, edge_trgs


    def sample_discrete_features_align(self, probX, node_mask):
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


    def apply_noise(self, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, node_mask, device, t_int=None):
        '''
        mask_transition: 64 x 19 x 19
        mask_align: 64 x 19 x 19
        align_matrix: 64 x 19 x 19
        node_mask: 64 x 19
        '''
        batch_num_nodes = torch.LongTensor(batch_num_nodes).to(device).reshape(-1, 1) # 64 x 1
        lowest_t = 0 if self.training else 1
        if t_int is None:
            t_int = torch.randint(lowest_t, self.T + 1, size=(batch_size,), device=device)
        else:
            t_int = torch.randint(t_int, t_int + 1, size=(batch_size,), device=device)

        s_int = t_int - 1

        t_float = t_int / self.T 
        s_float = s_int / self.T 

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float) 
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float) # this is okay!


        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, batch_num_nodes, batch_size=batch_size, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # Compute transition probabilities
        probX = align_matrix @ Qtb  # (bs, n, dx_out)

        sampled_t = self.sample_discrete_features_align(probX, node_mask)

        align_t = F.one_hot(sampled_t, num_classes=mask_transition.shape[1])

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'noise_align': align_t, 'mask_align': mask_align}

        return noisy_data


    def forward(self, noisy_data, s_mask, t_mask, graph_s_data, graph_t_data):
        return self.model(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)


    @torch.no_grad()
    def sample_batch(self, data, return_traj=False):
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
        trajectory = [x.clone()]
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size,)).float()
            t_array = s_array + 1
            s_norm = s_array / self.T 
            t_norm = t_array / self.T 

            sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, x, batch_num_nodes, mask_transition, mask_align, s_mask, t_mask, graph_s_data, graph_t_data)
            x = sampled_s
            trajectory.append(sampled_s.clone())

        if return_traj:
            return trajectory

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
        X_s = F.one_hot(sampled_s, num_classes=mask_align.shape[2]).float()
        return X_s
