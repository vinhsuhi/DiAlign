import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from models.bbgm_model import Net
from diffusion.noise_schedule import DiscreteUniformTransitionAlign, PredefinedNoiseScheduleDiscrete, MarginalUniMatchingTransition
from src.diffusion import diffusion_utils
from src.metrics.align_metrics import AlignAcc
from torch_geometric.utils import to_dense_batch
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import linear_sum_assignment



class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, val_names):
        super().__init__()
        self.cfg = cfg
        self.val_names = val_names
        self.name = cfg.general.name # graph-tf-model
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps # 50
        self.num_visual = cfg.model.num_visual # 8
        
        # if self.T < self.num_visual:
        #     print('Number of diffusion steps {} which is not enough to visual {} instances'.format(self.T, self.num_visual))
        #     print('So we only visualise {} instances instead'.format(self.T))
        #     self.step_visual = 1
        # else:
        #     self.step_visual = self.T // self.num_visual
        
        self.model = Net()
        
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps) # done

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransitionAlign()
        elif cfg.model.transition == 'matching':
            # TODO: revise this!
            self.transition_model = MarginalUniMatchingTransition(max_num_classes = 400) # @Vinh: Please replace this `400` with the actual `max_num_classes`
        else:
            raise NotImplementedError()

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
        val = S[y.bool()]
        nll = -torch.log(val + EPS)
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)


    def training_step(self, batch, batch_idx):
        '''
        done revised!
        '''
        update_info = self.gen_inter_info(batch)
        mask_transition = update_info['mask_transition']
        s_mask = update_info['s_mask']
        bs = update_info['bs']
        num_classes = update_info['num_classes']
        X0 = batch['gt_perm_mat'][0]
        noisy_data = self.apply_noise(X0, update_info)
        
        p_probX0 = self.forward(noisy_data, s_mask, batch, update_info, pad=True)
        loss_ce = self.loss_ce(p_probX0[s_mask], X0[s_mask], reduction='mean')
        
        if self.cfg.model.loss_type == 'hybrid':
            loss_lvb = self.loss_Lt(X0, p_probX0, noisy_data, s_mask, num_classes, bs, mask_transition)
            loss = self.ce_weight * loss_ce + self.vb_weight * loss_lvb 
            self.log('train/loss', loss, on_epoch=True, batch_size=X0.sum())
            self.log('train/loss_lvb', loss_lvb, on_epoch=True, batch_size=X0.sum())
        elif self.cfg.model.loss_type == 'ce': # only cross-entropy loss
            loss = loss_ce
            self.log('train/loss', loss, on_epoch=True, batch_size=X0.sum())
        elif self.cfg.model.loss_type == 'lvb_advance':
            print('This advanced loss has not been implemented, exitting...') 
            exit()
        else:
            print('This loss has not been implemented, exitting...!')
            exit()
        return {'loss': loss}


    def configure_optimizers(self):
        backbone_params = list(self.model.node_layers.parameters()) + list(self.model.edge_layers.parameters())
        backbone_params += list(self.model.final_layers.parameters())

        backbone_ids = [id(item) for item in backbone_params]

        new_params = [param for param in self.model.parameters() if id(param) not in backbone_ids]
        opt_params = [
            dict(params=backbone_params, lr=self.cfg.train.lr * 0.01),
            dict(params=new_params, lr=self.cfg.train.lr),
        ]
        return torch.optim.Adam(opt_params)

    
    def on_validation_start(self):
        print('Start evaluation...')


    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.current_epoch < 30:
            return None
        category = self.val_names[dataloader_idx]
        
        update_info = self.gen_inter_info(batch)
        s_mask = update_info['s_mask']
        X0 = batch['gt_perm_mat'][0]
        
        sample, pred = self.sample_batch(batch, update_info)
        test_acc = self.val_acc(pred, X0[s_mask])
        self.log("test/acc_epoch/{}".format(category), test_acc, batch_size=X0.sum().item())
        return {'test_acc': test_acc, 'bs': X0.sum().item()}


    def validation_epoch_end(self, outs):
        if self.current_epoch < 30:
            return None
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
        print('Evaluation finished!')


    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()
        self.test_acc.reset()
        print("Start evaluating after {} epochs...".format(self.current_epoch))


    def on_test_epoch_start(self) -> None:
        # print('Visualizing output...')

        # for batch in self.train_samples_to_visual:
        #     batch = batch.to(self.device)
        #     self.visualize_batch(batch, dtn='train')
        #     break

        # for batch in self.test_samples_to_visual:
        #     batch = batch.to(self.device)
        #     self.visualize_batch(batch, dtn='test')
        #     break

        # print('Done visualization!')
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
            try:
                accs.append(this_acc / this_sum)
            except:
                accs.append(this_acc)
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
            noisy_sample = self.apply_noise(X0, mask_align, mask_transition, bs, s_mask, t_mask, t_int=t_int)
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
            TODO: optimize this!    
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
        

    def sample_discrete_features_align_wr(self, q_probXt, s_mask):
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


    def apply_noise(self, X0, update_info, t_int=None):
        '''
        revised!
        '''
        bs = update_info['bs']
        num_classes = update_info['num_classes']
        mask_transition = update_info['mask_transition']
        s_mask = update_info['s_mask']
        t_mask = update_info['t_mask']
        mask_align = update_info['mask_align']
        
        
        lowest_t = 0 if self.training else 1
        if t_int is None:
            t_int = torch.randint(lowest_t, self.T + 1, size=(bs,), device=self.device)
        else:
            t_int = torch.randint(t_int, t_int + 1, size=(bs,), device=self.device)

        s_int = t_int - 1

        t_float = t_int / self.T 
        s_float = s_int / self.T 

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_sb = self.noise_schedule.get_alpha_bar(t_normalized=s_float) 
        alpha_tb = self.noise_schedule.get_alpha_bar(t_normalized=t_float) # this is okay!

        Qtb = self.transition_model.get_Qtb(alpha_tb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qtb[torch.isnan(Qtb)] = 0.1
        # Compute transition probabilities
        q_probXt = X0 @ Qtb  # (bs, n, dx_out) # keep this

        if self.cfg.model.sample_mode == 'wr':
            try:
                sampled_t = self.sample_discrete_features_align_wr(q_probXt, s_mask)
            except RuntimeError:
                import pdb; pdb.set_trace()
        elif self.cfg.model.sample_mode == 'wor':
            sampled_t = self.sample_discrete_features_align_wor(q_probXt, s_mask, t_mask)

        Xt = F.one_hot(sampled_t, num_classes=mask_transition.shape[1])

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_sb': alpha_sb,
                      'alpha_tb': alpha_tb, 'Xt': Xt, 'mask_align': mask_align}
        
        return noisy_data


    def forward(self, noisy_data, s_mask, data, update_info, pad=False):
        return self.model(noisy_data, s_mask, data, update_info, pad=pad)


    def sample_discrete_feature_noise_wor(self, bs, mask_align, s_mask, t_mask, lim_dist):
        '''
        TODO: optimize this function!
        bs: batch size
        mask_align: bs x n_s x n_t
        s_mask: bs x n_s
        t_mask: bs x n_t
        lim_dist: limit distribution
        '''
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


    def metropolis_hastings(self, prob):
        n_episodes = 20
        N = prob.shape[1] # N is the number of target nodes (which is greater or equal number of source nodes)
        n = prob.shape[0] # n is the number of source nodes

        # adding pseudo nodes to graph (so that we have equal N and n), and pseudo alignment prob!
        padded_prob = torch.zeros(N, N).cuda()
        padded_prob[:n,:] = prob
        padded_prob[n:, :] += 1 / N # uniform transition probs for pseudo nodes

        # initialize sigma, which is a random order of source nodes
        sigma = torch.randperm(N).cuda()
        # initialize pi
        pi = torch.zeros(N).long().cuda()
        # initialize mapping
        # source node `i` is mapped to target node `mapping[i]`
        mapping = torch.argsort(sigma)

        for _ in range(n_episodes): # until converge
            this_prob = padded_prob.clone()
            # step 1
            for i in range(N):
                this_prob_row = this_prob[sigma[i].item()]
                # re-normalize the prob:
                this_prob_row = (this_prob_row + 1e-12) / (this_prob_row + 1e-12).sum()
                # take a sample for the src node
                pi[i] = this_prob_row.multinomial(1).item() # pi[i] = j
                this_prob[:, pi[i]] = 0 # mask the sampled one
            
            this_prob = padded_prob.clone() # one instance from the batch
            # step 2
            log_prob_pi = torch.log(this_prob[sigma, pi]).sum()
            log_prob_sigma = torch.log(this_prob[sigma, torch.arange(N).cuda()]).sum()

            log_q_pi_given_sigma = torch.tensor(0.0).cuda()
            this_prob = padded_prob.clone()
            for i in range(N):
                # re-normalize the remaining probs
                this_prob = (this_prob + 1e-12) / (this_prob + 1e-12).sum(dim=1, keepdim=True)
                # take the probability of pair (sigma(i), pi(i))
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
            # print(mapping, acc_prob)
        return mapping, acc_prob


    def sample_discrete_feature_noise_wr(self, bs, lim_dist):
        sample = lim_dist.flatten(end_dim=-2).multinomial(1).reshape(bs, -1)
        sample = sample.long()
        sample = F.one_hot(sample, num_classes=lim_dist.shape[-1]).float()
        return sample


    def gen_inter_info(self, data):
        '''
        speed optimized! DONE!
        '''
        bs = len(data['gt_perm_mat'][0])
        num_classes = data['gt_perm_mat'][0].shape[-1]
        num_variables = data['gt_perm_mat'][0].shape[-2]
        
        src_graph, trg_graph = data['edges'][0], data['edges'][1]
        num_src_nodes = torch.bincount(src_graph.batch)
        num_trg_nodes = torch.bincount(trg_graph.batch)
        
        s_mask = torch.zeros(bs, num_variables).to(self.device).bool()
        t_mask = torch.zeros(bs, num_classes).to(self.device).bool()
        
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


    @torch.no_grad()
    def sample_batch(self, data, update_info, return_traj=False):
        '''
        get reverse samples from a batch of data
        '''
        mask_align = update_info['mask_align']
        mask_transition = update_info['mask_transition']
        s_mask = update_info['s_mask']
        t_mask = update_info['t_mask']
        bs = update_info['bs']
        num_classes = update_info['num_classes']
                
        lim_dist = mask_align / (mask_align.sum(dim=-1).reshape(bs, mask_align.shape[1], 1) + 1e-8)
        lim_dist[~s_mask] = 1 / lim_dist.shape[-1]
        
        if self.cfg.model.sample_mode == 'wr':
            XT = self.sample_discrete_feature_noise_wr(bs, lim_dist)
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
            Xs = self.sample_p_Xs_given_Xt(s_norm, t_norm, X, num_classes, mask_transition, mask_align, s_mask, t_mask, s_int, data, update_info)
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


    def sample_p_Xs_given_Xt(self, s, t, Xt, num_classes, 
                                mask_transition, mask_align, s_mask, t_mask, s_int, data, update_info):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = Xt.shape # dxs is the size of classes for each node, include padded ones

        beta_t = self.noise_schedule(t_normalized=t) # 
        alpha_sb = self.noise_schedule.get_alpha_bar(t_normalized=s) # 64 x 1
        alpha_tb = self.noise_schedule.get_alpha_bar(t_normalized=t) # 64 x 1

        Qtb = self.transition_model.get_Qtb(alpha_tb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb = self.transition_model.get_Qtb(alpha_sb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qtb[torch.isnan(Qtb)] = 0.1
        Qsb[torch.isnan(Qsb)] = 0.1
        if isinstance(self.transition_model, MarginalUniMatchingTransition):
            Qt = self.transition_model.get_Qt(beta_t, alpha_sb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        else:
            Qt = self.transition_model.get_Qt(beta_t, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)


        noisy_data = {'t': t, 'beta_t': beta_t, 'alpha_sb': alpha_sb,
                      'alpha_tb': alpha_tb, 'Xt': Xt, 'mask_align': mask_align} 

        # p(x_0 | x_t)
        p_probX0 = self.forward(noisy_data, s_mask, data, update_info, pad=True)
        
        # import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        # p_probX0, _ = to_dense_batch(p_probX0, , fill_value=0)

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
                # this_prob = prob[this_s_mask][:, this_t_mask]
                this_sampled_s = self.metropolis_hastings(prob)
                # this_sampled_s = self.sample_gumbel_noise(prob)[0]
                num_src_to_pad = len(this_s_mask) - len(this_sampled_s)
                if num_src_to_pad > 0:
                    this_sampled_s = np.concatenate((this_sampled_s, np.zeros(num_src_to_pad)))
                sampled_s.append(this_sampled_s)
            sampled_s = torch.LongTensor(np.stack(sampled_s)).to(self.device)
        elif self.cfg.model.sample_mode == 'wor' and (s_int > 0 or not self.cfg.model.use_argmax):
            sampled_s = self.sample_discrete_features_align_wor(p_probXs_givenXt, s_mask, t_mask) 
        elif self.cfg.model.sample_mode == 'wr' and (s_int > 0 or not self.cfg.model.use_argmax):
            sampled_s = self.sample_discrete_features_align_wr(p_probXs_givenXt, s_mask) 
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


    def loss_Lt(self, X0, p_probX0, noisy_data, s_mask, num_classes, bs, mask_transition):

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
        alpha_tb = noisy_data['alpha_tb']
        alpha_sb = noisy_data['alpha_sb']
        beta_t = noisy_data['beta_t']
        Xt = noisy_data['Xt']
        Qtb = self.transition_model.get_Qtb(alpha_tb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb = self.transition_model.get_Qtb(alpha_sb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qtb[torch.isnan(Qtb)] = 0.1
        Qsb[torch.isnan(Qsb)] = 0.1
        if isinstance(self.transition_model, MarginalUniMatchingTransition):
            Qt = self.transition_model.get_Qt(beta_t, alpha_sb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        else:
            Qt = self.transition_model.get_Qt(beta_t, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        # q(x_{t-1} | x_t, x_0)
        posterior_true = diffusion_utils.compute_posterior_distribution(X0, Xt, Qt, Qsb, Qtb)
        # p(x_{t-1} | x_t)
        posterior_pred = diffusion_utils.compute_posterior_distribution(p_probX0, Xt, Qt, Qsb, Qtb)

        # Reshape and filter masked rows
        posterior_true, posterior_pred = diffusion_utils.mask_distributions_align(posterior_true, posterior_pred, s_mask)

        return self.kl_divergence_with_probs(posterior_true, posterior_pred).mean()

