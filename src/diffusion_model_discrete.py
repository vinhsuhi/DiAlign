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
import numpy as np 
from scipy.optimize import linear_sum_assignment

def generate_y(y_col, device):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.general.name # graph-tf-model
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps # 50
        self.num_visual = cfg.model.num_visual # 8
        if self.T < self.num_visual:
            print('Number of diffusion steps {} which is not enough to visual {} instances'.format(self.T, self.num_visual))
            print('So we only visualise {} instances instead'.format(self.T))
            self.step_visual = 1
        else:
            self.step_visual = self.T // self.num_visual
        self.model = GraphTransformerMatching(scalar_dim=cfg.model.scalar_dim, num_layers=cfg.model.num_layers, 
                                              ori_feat_dim=cfg.dataset.ori_dim, embed_dim=cfg.model.embed_dim, cat=cfg.model.cat, lin=cfg.model.lin, dropout=cfg.model.dropout, use_time=cfg.model.use_time)
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps) # done

        #if cfg.model.transition == 'uniform':
        self.transition_model = DiscreteUniformTransitionAlign()

        # log hyperparameters
        self.ce_weight = cfg.model.ce_weight
        self.vb_weight = cfg.model.vb_weight
        self.save_hyperparameters()

        self.train_acc = AlignAcc()
        self.val_acc = AlignAcc()
        self.test_acc = AlignAcc()
        self.log_every_steps = cfg.general.log_every_steps

        self.train_samples_to_visual = None 
        self.test_samples_to_visual = None


    def loss_ce(self, S, y, reduction='mean'):
        '''
        Return the cross entropy loss
        '''
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
        '''
        Extract element, padding, create masks from a batch of data
        '''
        x_src, x_trg = batch.x_s, batch.x_t
        edge_index_s, edge_index_t = batch.edge_index_s, batch.edge_index_t
        edge_attr_s, edge_attr_t = batch.edge_attr_s, batch.edge_attr_t 
        name_s, name_t = batch.name_s, batch.name_t
        batch_s, batch_t = batch.x_s_batch, batch.x_t_batch

        x_src_, s_mask = to_dense_batch(x_src, batch_s, fill_value=0)
        x_trg_, t_mask = to_dense_batch(x_trg, batch_t, fill_value=0)
        bs = x_src_.shape[0]

        graph_s_data = {'x': x_src, 'edge_index': edge_index_s, 'edge_attr': edge_attr_s, 'name': name_s, 'batch': batch_s}
        graph_t_data = {'x': x_trg, 'edge_index': edge_index_t, 'edge_attr': edge_attr_t, 'name': name_t, 'batch': batch_t}

        mask_align = s_mask.unsqueeze(2) * t_mask.unsqueeze(1)
        mask_transition = t_mask.unsqueeze(2) * t_mask.unsqueeze(1)

        X0 = torch.zeros((bs, s_mask.shape[1], t_mask.shape[1])).to(self.device)
        
        batch_num_nodes = list()
        for j in range(bs):
            this_y = generate_y(batch[j].y, device=self.device)
            X0[j][this_y[0], this_y[1]] = 1.
            batch_num_nodes.append(len(batch[j].x_t))

        batch_num_nodes = torch.LongTensor(batch_num_nodes).to(self.device).reshape(-1, 1) # 64 x 1

        return x_src, x_trg, X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data


    def training_step(self, batch, batch_idx):
        '''
        where we add noise to clean data and ask the model to reconstruct it
        we also implement several types of losses here!
        '''
        _, _, X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data = self.sub_forward(batch)
        noisy_data = self.apply_noise(X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, self.device)
        p_probX0 = self.forward(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)
        p_probX0_batch, _ = to_dense_batch(p_probX0, graph_s_data['batch'], fill_value=0)
        target = generate_y(batch.y, device=self.device)
        loss_ce = self.loss_ce(p_probX0, target)
        if self.cfg.model.loss_type == 'hybrid':
            loss_lvb = self.loss_Lt(X0, p_probX0_batch, noisy_data, s_mask, batch_num_nodes, bs, mask_transition)
            loss = self.ce_weight * loss_ce + self.vb_weight * loss_lvb 
            self.log('train/loss', loss, on_epoch=True, batch_size=target.shape[1])
            self.log('train/loss_lvb', loss_lvb, on_epoch=True, batch_size=target.shape[1])
        elif self.cfg.model.loss_type == 'ce': # only cross-entropy loss
            loss = loss_ce
            self.log('train/loss', loss, on_epoch=True, batch_size=target.shape[1])
        elif self.cfg.model.loss_type == 'lvb_advance':
            print('This advanced loss has not been implemented, exitting...') 
            exit()
        else:
            print('This loss has not been implemented, exitting...!')
            exit()
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


    def kl_prior(self, gold_align, s_mask, mask_transition):
        """Computes the KL between q(xT | x) and the prior p(xT)

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((gold_align.size(0), 1), device=self.device)
        Ts = self.T * ones
        alpha_tb = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1) okay this is correct!
        Qtb = self.transition_model.get_Qtb(alpha_tb, self.device) * mask_transition

        # Compute transition probabilities
        prob_align = gold_align @ Qtb  # (bs, n, dx_out) q(X_T)

        bs, n, _ = prob_align.shape # n la so nodes

        lim_dist = mask_align / (mask_align.sum(dim=-1).reshape(batch_size, mask_align.shape[1], 1) + 1e-8)
        # Make sure that masked rows do not contribute to the loss
        limit_dist_align, prob_align = diffusion_utils.mask_distributions_align(true_align=limit_dist.clone(), pred_align=prob_align, s_mask=s_mask)

        kl_distance_align = F.kl_div(input=prob_align.log(), target=limit_dist_align, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_align)


    # def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
    #     """Computes an estimator for the variational lower bound.
    #        pred: (batch_size, n, total_features)
    #        noisy_data: dict
    #        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy) clean data
    #        node_mask : (bs, n)
    #        Output: nll (size 1)
    #    """
    #     t = noisy_data['t']

    #     # 1. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
    #     kl_prior = self.kl_prior(X, E, node_mask) # done

    #     # 2. Diffusion loss
    #     loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

    #     # 3. Reconstruction loss
    #     # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    #     prob0 = self.reconstruction_logp(t, X, E, node_mask)

    #     loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

    #     # Combine terms
    #     nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
    #     assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

    #     # Update NLL metric object and return batch nll
    #     nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

    #     wandb.log({"kl prior": kl_prior.mean(),
    #                "Estimator loss terms": loss_all_t.mean(),
    #                "log_pn": log_pN.mean(),
    #                "loss_term_0": loss_term_0,
    #                'batch_test_nll' if test else 'val_nll': nll}, commit=False)
    #     return nll


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
        _, _, align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, _, _ = self.sub_forward(batch)
        poses, gts, edge_srcs, edge_trgs = self.forward_diffusion(align_matrix, mask_align, mask_transition, batch_size, batch_num_nodes, s_mask, t_mask, batch, dtn=dtn)
        self.reverse_diffusion(batch, poses, gts, edge_srcs, edge_trgs, s_mask, t_mask, dtn=dtn)

    
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
    

    def reverse_diffusion(self, batch, poses, gts, edge_srcs, edge_trgs, s_mask, t_mask, dtn='train'):
        device = self.device
        trajectory = self.sample_batch(batch, return_traj=True)
        for s_int in reversed(range(0, self.T + 1)):
            align = trajectory[s_int]
            for i in range(len(align)): # number of data examples to be visualised
                this_align = generate_y(align[i][s_mask[i]].argmax(dim=1), device)
                if ((self.T - s_int) % self.step_visual == 0) or (s_int == 0):
                    path_visual = 'visuals/{}/sp{}/X_pred_{}.png'.format(dtn, i, self.T - s_int)
                    self.visualise_instance(edge_srcs[i], edge_trgs[i], this_align, path_visual, poses[i])


    def forward_diffusion(self, X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, batch, dtn='train'):
        device = self.device
        # draw the first no-noise graph

        gts = list()
        edge_srcs = list()
        edge_trgs = list()
        poses = list()
        for i in range(bs):
            this_batch = batch[i]
            edge_srcs.append(this_batch['edge_index_s'])
            edge_trgs.append(this_batch['edge_index_t'])
            gts.append(generate_y(this_batch.y, device))
            path_visual = 'visuals/{}/sp{}/'.format(dtn, i)
            if not os.path.exists(path_visual):
                os.makedirs(path_visual)
            poses.append(self.visualise_instance(edge_srcs[-1], edge_trgs[-1], gts[-1], '{}/X_0.png'.format(path_visual)))

        for t_int in range(0, self.T):
            noisy_sample = self.apply_noise(X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, device, t_int=t_int)
            Xt = noisy_sample['Xt']
            for i in range(bs):
                this_noisy_alignment = generate_y(Xt[i][s_mask[i]].argmax(dim=1), device)
                if ((t_int + 1) % self.step_visual == 0) or (t_int + 1) == self.T:
                    path_visual = 'visuals/{}/sp{}/X_{}.png'.format(dtn, i, t_int + 1)
                    self.visualise_instance(edge_srcs[i], edge_trgs[i], this_noisy_alignment, path_visual, poses[i])
        return poses, gts, edge_srcs, edge_trgs


    def sample_discrete_features_align_wor(self, q_probXt, s_mask, t_mask):
        ''' Sample features from multinomial distribution (w/o replacement) with given probabilities
            :param q_probXt: bs, n, dx_out        
        '''
        bs, n, _ = q_probXt.shape
        # Noise X
        # The masked rows should define probability distributions as well
        q_probXt[~s_mask] = 1 / q_probXt.shape[-1]

        # Flatten the probability tensor to sample with multinomial

        # Sample X
        samples = list()
        for i in range(len(q_probXt)): # iterate over batch samples
            this_probX = q_probXt[i].clone()
            this_s_mask = s_mask[i]
            this_t_mask = t_mask[i]
            num_nodes_src = this_s_mask.sum().item()
            num_nodes_trg = this_t_mask.sum().item()
            random_indices_src = torch.randperm(num_nodes_src).to(self.device) # keep this
             
            srcs = list()
            trgs = list()
            for j in range(len(random_indices_src)):
                src_node = random_indices_src[j].item()
                src_prob = this_probX[src_node] / this_probX[src_node].sum()
                trg_node = src_prob.multinomial(1).item()
                this_probX[:, trg_node] = 0
                srcs.append(src_node)
                trgs.append(trg_node)

            srcs = torch.LongTensor(srcs).to(self.device)
            trgs = torch.LongTensor(trgs).to(self.device)

            if this_probX.shape[0] > num_nodes_src:
                left_over_src = torch.arange(num_nodes_src, this_probX.shape[0]).to(self.device)
                left_over_trg = [this_probX.shape[1] - 1] * len(left_over_src)
                left_over_trg = torch.LongTensor(left_over_trg).to(self.device)
                srcs = torch.cat((srcs, left_over_src))
                trgs = torch.cat((trgs, left_over_trg))
            
            samples.append(trgs[srcs.sort()[1]])
        Xt = torch.stack(samples)
        return Xt



    def sample_by_greedy(self, q_probXt, s_mask, t_mask):
        bs, n, _ = q_probXt.shape 

        q_probXt[~s_mask] = 0
        samples = list() 
        for b in range(bs):
            # srcs = list()
            # trgs = list()
            # this_probXt = q_probXt[b]
            # for i in range(n):
            #     argm = torch.argmax(this_probXt).item()
            #     src = argm // this_probXt.shape[1]
            #     trg = argm % this_probXt.shape[1]
            #     srcs.append(src)
            #     trgs.append(trg)
            #     this_probXt[src] = 0
            #     this_probXt[:, trg] = 0
            # srcs = torch.LongTensor(srcs).to(self.device)
            # trgs = torch.LongTensor(trgs).to(self.device)
            # samples.append(trgs[srcs.sort()[1]])

            this_probXt = q_probXt[b].detach().cpu().numpy() # cost matrix
            srcs, trgs = linear_sum_assignment(1 - this_probXt)
            samples.append(torch.LongTensor(trgs[srcs.sort()]).to(self.device))
            #import pdb; pdb.set_trace()
        Xt = torch.cat(samples, dim=0)
        return Xt    
        


    def sample_discrete_features_align_wr(self, q_probXt, s_mask, t_mask):
        ''' Sample features from multinomial distribution with given probabilities q_probXt
            :param probX: bs, n, dx_out        node features
            :param proby: bs, dy_out           global features.
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


    def apply_noise(self, X0, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, device, t_int=None):
        '''
        mask_transition: 64 x 19 x 19
        mask_align: 64 x 19 x 19
        align_matrix: 64 x 19 x 19
        s_mask: 64 x 19
        '''
        
        lowest_t = 0 if self.training else 1
        if t_int is None:
            t_int = torch.randint(lowest_t, self.T + 1, size=(bs,), device=device)
        else:
            t_int = torch.randint(t_int, t_int + 1, size=(bs,), device=device)

        s_int = t_int - 1

        t_float = t_int / self.T 
        s_float = s_int / self.T 

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_sb = self.noise_schedule.get_alpha_bar(t_normalized=s_float) 
        alpha_tb = self.noise_schedule.get_alpha_bar(t_normalized=t_float) # this is okay!

        Qtb = self.transition_model.get_Qtb(alpha_tb, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # Compute transition probabilities
        q_probXt = X0 @ Qtb  # (bs, n, dx_out) # keep this

        if self.cfg.model.sample_mode == 'wr':
            sampled_t = self.sample_discrete_features_align_wr(q_probXt, s_mask, t_mask)
        elif self.cfg.model.sample_mode == 'wor':
            sampled_t = self.sample_discrete_features_align_wor(q_probXt, s_mask, t_mask)

        Xt = F.one_hot(sampled_t, num_classes=mask_transition.shape[1])

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_sb': alpha_sb,
                      'alpha_tb': alpha_tb, 'Xt': Xt, 'mask_align': mask_align}

        return noisy_data


    def forward(self, noisy_data, s_mask, t_mask, graph_s_data, graph_t_data):
        return self.model(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)


    def sample_discrete_feature_noise_wor(self, bs, mask_align, s_mask, t_mask, lim_dist):
        sample = list()
        for i in range(bs):
            mask_align_i = mask_align[i]
            this_s_mask = s_mask[i]
            this_t_mask = t_mask[i]
            num_nodes_src = this_s_mask.sum().item()
            num_nodes_trg = this_t_mask.sum().item()
            random_indices_src = torch.randperm(num_nodes_src).to(self.device)
            random_indices_trg = torch.randperm(num_nodes_trg).to(self.device)
            trg_assigned_to_src = random_indices_trg[:num_nodes_src]

            if (mask_align_i.shape[0] > num_nodes_src):
                left_over_trg = random_indices_trg[num_nodes_src:]
                if mask_align_i.shape[1] > num_nodes_trg:
                    left_over_trg = torch.cat((left_over_trg, torch.arange(num_nodes_trg, mask_align_i.shape[1]).to(self.device)))
                left_over_src = torch.arange(num_nodes_src, mask_align_i.shape[0]).to(self.device)
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


    def sample_discrete_features_align_metropolis(self, prob):
        ''' Sample features from multinomial distribution (w/o replacement) with given probabilities
            :param q_probXt: bs, n, dx_out        
        '''
        bs, n, _ = q_probXt.shape
        # Noise X
        # The masked rows should define probability distributions as well
        #q_probXt[~s_mask] = 1 / q_probXt.shape[-1]
        # Flatten the probability tensor to sample with multinomial
        # Sample X
        
        n_episodes = 20
        prob = probs[0].clone() # one instance from the batch
        N = prob.shape[1] # N is the number of target nodes (which is greater or equal number of source nodes)
        n = prob.shape[0] # n is the number of source nodes

        # adding pseudo nodes to graph (so that we have equal N and n), and pseudo alignment prob!
        padded_prob = torch.zeros(N, N)
        padded_prob[:n,:] = prob
        padded_prob[n:, :] += 1 / N # uniform transition probs for pseudo nodes

        # initialize sigma, which is a random order of source nodes
        sigma = torch.randperm(N).to(self.device)
        # initialize pi
        pi = torch.zeros(N).long().to(self.device)

        for _ in range(n_episodes): # until converge
            this_prob = padded_prob.clone()
            # step 1
            for i in range(N):
                pi[i] = this_prob[sigma[i].item()].multinomial(1).item() # pi[i] = j
                this_prob[:, trg_node] = 0 # mask the sampled one
                # re-normalize the probability matrix
                this_prob = (this_prob + 1e-12) / (this_prob + 1e-12).sum(dim=1, keepdim=True)
            
            this_prob = padded_prob.clone() # one instance from the batch
            # step 2
            prob_pi = this_prob[sigma, pi].prod()
            prob_sigma = this_prob[torch.arange(N).to(self.device), sigma]

            q_pi_given_sigma = torch.zeros(N).to(self.device) # initialize the prob
            for i in range(N):
                if i == 0:
                    continue # there is no edges that have been taken
                this_prob[:, pi[:i]] = 0
                # re-normalize the remaining probs
                this_prob = (this_prob + 1e-12) / (this_prob + 1e-12).sum(dim=1, keepdim=True)
                # take the probability of pair (pi(i), sigma(i))
                q_pi_given_sigma[i] = this_prob[pi[i], sigma[i]]
            q_pi_given_sigma = q_pi_given_sigma.prod()
            
            q_sigma_given_pi = torch.zeros(N).to(self.device) # initialize the prob
            for i in range(N):
                if i == 0:
                    continue # there is no edges that have been taken
                this_prob[:, sigma[:i]] = 0
                # re-normalize the remaining probs
                this_prob = (this_prob + 1e-12) / (this_prob + 1e-12).sum(dim=1, keepdim=True)
                # take the probability of pair (pi(i), sigma(i))
                q_sigma_given_pi[i] = this_prob[sigma[i], pi[i]]
            q_sigma_given_pi = q_sigma_given_pi.prod()
            
            # if Uniform(0,1) < p(π)·q(σ|π) / (p(σ)·q(π|σ))
            if np.random.rand() < prob_pi * q_sigma_given_pi / (prob_sigma * q_pi_given_sigma):
                # accept new sigma
                sigma = pi[sigma]
        return Xt


    def sample_discrete_feature_noise_wr(self, bs, mask_align, s_mask, t_mask, lim_dist):
        sample = lim_dist.flatten(end_dim=-2).multinomial(1).reshape(bs, -1)
        long_mask = s_mask.long()
        sample = sample.type_as(long_mask)
        sample = F.one_hot(sample, num_classes=lim_dist.shape[-1]).float()
        return sample


    @torch.no_grad()
    def sample_batch(self, data, return_traj=False):
        _, _, _, mask_align, mask_transition, bs, batch_num_nodes, s_mask, t_mask, graph_s_data, graph_t_data = self.sub_forward(data)
        lim_dist = mask_align / (mask_align.sum(dim=-1).reshape(bs, mask_align.shape[1], 1) + 1e-8)
        lim_dist[~s_mask] = 1 / lim_dist.shape[-1]
        
        if self.cfg.model.sample_mode == 'wr':
            XT = self.sample_discrete_feature_noise_wr(bs, mask_align, s_mask, t_mask, lim_dist)
        elif self.cfg.model.sample_mode == 'wor':
            XT = self.sample_discrete_feature_noise_wor(bs, mask_align, s_mask, t_mask, lim_dist)
        else:
            print('This prior sampling has not been implemented, exitting...')
            exit()

        X = XT
        trajectory = [X.clone()]
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((bs,)).float()
            t_array = s_array + 1
            s_norm = s_array / self.T 
            t_norm = t_array / self.T 

            Xs = self.sample_p_Xs_given_Xt(s_norm, t_norm, X, batch_num_nodes, mask_transition, mask_align, s_mask, t_mask, graph_s_data, graph_t_data, s_int)
            X = Xs
            trajectory.append(Xs.clone())

        if return_traj:
            return trajectory

        collapsed_X = X[s_mask]
        return X, collapsed_X

    def sample_gumbel_noise(self, prob):
        '''
        prob: the posterior of size (n_src_nodes x n_trg_nodes)
        '''
        from torch.distributions.gumbel import Gumbel
        log_prob = torch.log(prob)
        # generate gumbel noise Gumbel(0, 1)
        gumbel_distribution = Gumbel(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
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

    def sample_p_Xs_given_Xt(self, s, t, Xt, batch_num_nodes, 
                                mask_transition, mask_align, s_mask, t_mask, graph_s_data, graph_t_data, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = Xt.shape # dxs is the size of classes for each node, include padded ones

        beta_t = self.noise_schedule(t_normalized=t) # 
        alpha_sb = self.noise_schedule.get_alpha_bar(t_normalized=s) # 64 x 1
        alpha_tb = self.noise_schedule.get_alpha_bar(t_normalized=t) # 64 x 1

        Qtb = self.transition_model.get_Qtb(alpha_tb, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb = self.transition_model.get_Qtb(alpha_sb, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qt = self.transition_model.get_Qt(beta_t, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)


        noisy_data = {'t': t, 'beta_t': beta_t, 'alpha_sb': alpha_sb,
                      'alpha_tb': alpha_tb, 'Xt': Xt, 'mask_align': mask_align} 

        # p(x_0 | x_t)
        p_probX0 = self.forward(noisy_data, s_mask, t_mask, graph_s_data, graph_t_data)
        p_probX0, _ = to_dense_batch(p_probX0, graph_s_data['batch'], fill_value=0)

        # q(x_{t-1}|x_t, x_0)
        q_Xs_given_Xt_and_X0 =  diffusion_utils.compute_batched_over0_posterior_distribution(X_t=Xt,
                                                                                            Qt=Qt,
                                                                                            Qsb=Qsb,
                                                                                            Qtb=Qtb) 

        # sum_x (q(x_{t-1} | x_t, x) * p(x_0 | x_t))
        weighted_X =  q_Xs_given_Xt_and_X0 * p_probX0.unsqueeze(-1)        # bs, n, d0, d_t-1
        unnormalized_probX = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_probX[torch.sum(unnormalized_probX, dim=-1) == 0] = 1e-5

        # p(x_{t-1} | x_t)
        p_probXs_givenXt = unnormalized_probX / torch.sum(unnormalized_probX, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((p_probXs_givenXt.sum(dim=-1) - 1).abs() < 1e-4).all()
        

        if self.cfg.model.metropolis and (s_int > 0 or not self.cfg.model.use_argmax):
            #sampled_s = self.sample_discrete_features_align_metropolis(p_probXs_givenXt, s_mask, t_mask) 
            sampled_s = list()
            for batch_idx in range(bs):
                prob = p_probXs_givenXt[batch_idx].clone()
                this_s_mask = s_mask[batch_idx]
                this_t_mask = t_mask[batch_idx]
                this_prob = prob[this_s_mask][:, this_t_mask]
                this_sampled_s = self.sample_gumbel_noise(prob)[0]
                num_src_to_pad = len(this_s_mask) - len(this_sampled_s)
                if num_src_to_pad > 0:
                    this_sampled_s = np.concatenate((this_sampled_s, np.zeros(num_src_to_pad)))
                sampled_s.append(this_sampled_s)
            sampled_s = torch.LongTensor(np.stack(sampled_s)).to(self.device)
        elif self.cfg.model.sample_mode == 'wor' and (s_int > 0 or not self.cfg.model.use_argmax):
            sampled_s = self.sample_discrete_features_align_wor(p_probXs_givenXt, s_mask, t_mask) 
        elif self.cfg.model.sample_mode == 'wr' and (s_int > 0 or not self.cfg.model.use_argmax):
            sampled_s = self.sample_discrete_features_align_wr(p_probXs_givenXt, s_mask, t_mask) 
        else:
            sampled_s = self.sample_by_greedy(p_probXs_givenXt, s_mask, t_mask)

        Xs = F.one_hot(sampled_s, num_classes=mask_align.shape[2]).float()
        return Xs


    def kl_divergence_with_probs(self, p = None, q = None, epsilon = 1e-20):
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


    def loss_Lt(self, X0, p_probX0, noisy_data, s_mask, batch_num_nodes, bs, mask_transition):

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
            batch_num_nodes
            bs: batch_size
            mask_translation
        '''
        alpha_tb = noisy_data['alpha_tb']
        alpha_sb = noisy_data['alpha_sb']
        beta_t = noisy_data['beta_t']
        Xt = noisy_data['Xt']
        Qtb = self.transition_model.get_Qtb(alpha_tb, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb = self.transition_model.get_Qtb(alpha_sb, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qt = self.transition_model.get_Qt(beta_t, batch_num_nodes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        # q(x_{t-1} | x_t, x_0)
        posterior_true = diffusion_utils.compute_posterior_distribution(X0, Xt, Qt, Qsb, Qtb)
        # p(x_{t-1} | x_t)
        posterior_pred = diffusion_utils.compute_posterior_distribution(p_probX0, Xt, Qt, Qsb, Qtb)

        # Reshape and filter masked rows
        posterior_true, posterior_pred = diffusion_utils.mask_distributions_align(posterior_true, posterior_pred, s_mask)

        return self.kl_divergence_with_probs(posterior_true, posterior_pred).mean()

