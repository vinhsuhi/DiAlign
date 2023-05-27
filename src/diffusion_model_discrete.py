import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from models.bbgm_model import Net
from diffusion.noise_schedule import DiscreteUniformTransitionAlign, PredefinedNoiseScheduleDiscrete, MarginalUniMatchingTransition
from src.diffusion import diffusion_utils
from src.metrics.align_metrics import AlignAcc
from src.losses.losses import loss_ce, loss_lvb_t
from torch_geometric.utils import to_dense_batch
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np 
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import os 


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
        
        self.model = Net(complex=False)
        
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps) # done

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransitionAlign()
        elif cfg.model.transition == 'matching':
            self.transition_model = MarginalUniMatchingTransition(max_num_classes = 400) # @Vinh: Please replace this `400` with the actual `max_num_classes`
        else:
            raise NotImplementedError()

        # log hyperparameters
        self.ce_weight = cfg.model.ce_weight
        self.vb_weight = cfg.model.vb_weight
        self.save_hyperparameters()

        self.train_acc, self.val_acc, self.test_acc = AlignAcc(), AlignAcc(), AlignAcc()
        self.log_every_steps = cfg.general.log_every_steps

        self.train_samples_to_visual = None 
        self.test_samples_to_visual = None
        
    
    def training_step(self, batch, batch_idx):
        self.model.without_diff = self.without_diff
        update_info = diffusion_utils.gen_inter_info(batch, self.device)
        s_mask = update_info['s_mask']
        X0 = batch['gt_perm_mat'][0]
        # read from here!
        noisy_data = self.apply_noise(X0, update_info)
        p_probX0 = self.model.forward(noisy_data, batch, update_info, pad=True, train=True)
        ce_loss = loss_ce(p_probX0[s_mask], X0[s_mask], reduction='mean')    
        if self.cfg.model.loss_type == 'hybrid':
            loss_lvb = loss_lvb_t(X0, p_probX0, noisy_data, update_info)
            loss = self.ce_weight * ce_loss + self.vb_weight * loss_lvb
            self.log('train/loss_lvb', loss_lvb, on_epoch=True, batch_size=X0.sum())
        elif self.cfg.model.loss_type == 'ce': # only cross-entropy loss
            loss = ce_loss
        self.log('train/loss', loss, on_epoch=True, batch_size=X0.sum())
        return {'loss': loss}


    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.model.without_diff = self.without_diff
        category = self.val_names[dataloader_idx]
        update_info = diffusion_utils.gen_inter_info(batch, self.device)
        s_mask = update_info['s_mask']
        X0 = batch['gt_perm_mat'][0]

        if self.without_diff:
            noisy_data = self.apply_noise(X0, update_info)
            prob = self.model.forward(noisy_data, batch, update_info, pad=True)
            sampled_s = diffusion_utils.sample_by_hungarian(prob, s_mask, update_info['t_mask'], self.device)
            pred = F.one_hot(sampled_s, num_classes=update_info['mask_align'].shape[2]).float()[s_mask]
        else:
            sample, pred, probX0s = self.sample_batch(batch, update_info)
        
        test_acc = self.val_acc(pred, X0[s_mask])
        self.log("test/acc_epoch/{}".format(category), test_acc, batch_size=X0.sum().item())
        return_dict = {'test_acc': test_acc.item(), 'bs': X0.sum().item()}
        return return_dict
    
    
    def apply_noise(self, X0, update_info, t_int=None, pt=None):
        '''
        revised!
        pt is a tensor of size T is for important sampling!
        '''
        s_mask, t_mask = update_info['s_mask'], update_info['t_mask']
        mask_transition = update_info['mask_transition']
        if t_int is None:
            lowest_t = 0 if self.training else 1
            t_int = torch.randint(lowest_t, self.T + 1, size=(update_info['bs'],), device=self.device)
        else:
            t_int = torch.randint(t_int, t_int + 1, size=(update_info['bs'],), device=self.device)

        s_int = t_int - 1
        t_float = t_int / self.T 
        s_float = s_int / self.T 
        
        transition_dict = self.get_transition_params(t_float, s_float, update_info['bs'], update_info['num_classes'], update_info['mask_transition'])
        
        # Compute transition probabilities
        q_probXt = X0 @ transition_dict['Qtb']  # (bs, n, dx_out) # keep this

        if self.cfg.model.sample_mode == 'wr':
            sampled_t = diffusion_utils.sample_discrete_features_align_wr(q_probXt, s_mask)
        else:
            sampled_t = diffusion_utils.sample_discrete_features_align_wor(q_probXt, s_mask, t_mask, self.device)

        Xt = F.one_hot(sampled_t, num_classes=update_info['mask_transition'].shape[1])

        noisy_data = {'t_int': t_int, 't': t_float,'Xt': Xt}
        noisy_data.update(transition_dict)
        return noisy_data


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
        X0 = data['gt_perm_mat'][0]
                
        lim_dist = mask_align / (mask_align.sum(dim=-1).reshape(bs, mask_align.shape[1], 1) + 1e-8)
        lim_dist[~s_mask] = 1 / lim_dist.shape[-1]
        
        if self.cfg.model.sample_mode == 'wr':
            XT = diffusion_utils.sample_discrete_feature_noise_wr(bs, lim_dist)
        else:
            XT = diffusion_utils.sample_discrete_feature_noise_wor(bs, mask_align, s_mask, t_mask, lim_dist, self.device)
            
        X = XT
        trajectory = [X.clone()]
        probX0s = list()
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((bs,)).float()
            t_array = s_array + 1
            s_norm = s_array / self.T 
            t_norm = t_array / self.T 
            
            Xs, prob_Xs_given_Xt, p_probX0, q_Xs_given_Xt_and_X0, noisy_data = self.sample_p_Xs_given_Xt(data, s_norm, t_norm, X, update_info)
            # evaluiate p_probX0 here
            probX0s.append(p_probX0)
            
            if self.cfg.general.test_only is not None and 0: 
                self.visualise_dist(bs, s_mask, t_mask, X, prob_Xs_given_Xt, p_probX0, q_Xs_given_Xt_and_X0, Qt, X0, s_int)
                       
            X = Xs
            trajectory.append(Xs.clone())
        
        if self.cfg.general.test_only is not None and 0:
            import pdb; pdb.set_trace()

        if return_traj:
            return trajectory

        collapsed_X = X[s_mask]
        return X, collapsed_X, probX0s


    def sample_p_Xs_given_Xt(self, data, s, t, Xt, update_info):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = Xt.shape # dxs is the size of classes for each node, include padded ones
        transition_dict = self.get_transition_params(t, s, bs, update_info['num_classes'], update_info['mask_transition'])
        noisy_data = {'t': t, 'Xt': Xt, 'mask_align': update_info['mask_align']}
        noisy_data.update(transition_dict) 

        # p(x_0 | x_t)
        p_probX0 = self.model.forward(noisy_data, data, update_info, pad=True) 
        # q(x_{t-1}|x_t, x_0)
        q_Xs_given_Xt_and_X0 =  diffusion_utils.compute_batched_over0_posterior_distribution(X_t=Xt, Qt=transition_dict['Qt'], Qsb=transition_dict['Qsb'],  Qtb=transition_dict['Qtb']) 
        
        posterior_true = diffusion_utils.compute_posterior_distribution(data['gt_perm_mat'][0], Xt, transition_dict['Qt'], transition_dict['Qsb'], transition_dict['Qtb'])
        
        # sum_x (q(x_{t-1} | x_t, x) * p(x_0 | x_t))
        weighted_X =  q_Xs_given_Xt_and_X0 * p_probX0.unsqueeze(-1)         # bs, n, d0, d_t-1
        unnormalized_probX = weighted_X.sum(dim=2)                          # bs, n, d_t-1
        unnormalized_probX[torch.sum(unnormalized_probX, dim=-1) == 0] = 1e-5

        # p(x_{t-1} | x_t)
        p_probXs_givenXt = unnormalized_probX / torch.sum(unnormalized_probX, dim=-1, keepdim=True)  # bs, n, d_t-1

        if not ((p_probXs_givenXt.sum(dim=-1) - 1).abs() < 1e-4).all():
            print('detected weird distribution')
            import pdb; pdb.set_trace()
        
        t_int = t[0].item() * self.T 
        s_int = t_int - 1
        s_mask = update_info['s_mask']
        
        # sampled_s = diffusion_utils.sample_by_hungarian(p_probXs_givenXt, s_mask, t_mask, self.device)
            
        if self.cfg.model.sample_mode=='perturb' and s_int > 0:
            sampled_s = self.perturb_sampling(p_probXs_givenXt, s_mask, t_mask)
        elif self.cfg.model.sample_mode == 'metropolis' and s_int > 0:
            sampled_s = self.metropolis_hastings(p_probXs_givenXt, s_mask, t_mask, n_episodes=20)
        elif self.cfg.model.sample_mode == 'wor' and s_int > 0:
            sampled_s = diffusion_utils.sample_discrete_features_align_wor(p_probXs_givenXt, s_mask, t_mask, self.device) 
        elif self.cfg.model.sample_mode == 'wr' and s_int > 0:
            sampled_s = diffusion_utils.sample_discrete_features_align_wr(p_probXs_givenXt, s_mask) 
        else: # s_int = 0
            sampled_s = diffusion_utils.sample_by_hungarian(p_probXs_givenXt, s_mask, t_mask, self.device)

        Xs = F.one_hot(sampled_s, num_classes=update_info['mask_align'].shape[2]).float()
        return Xs, p_probXs_givenXt, p_probX0, posterior_true, noisy_data


    def metropolis_hastings(self, p_probXs_givenXt, s_mask, t_mask, n_episodes):
        'has not moved to other files because it should be checked more'
        bs = len(s_mask)
        sampled_s = list()
        for batch_idx in range(bs):
            prob = p_probXs_givenXt[batch_idx].clone()
            this_s_mask = s_mask[batch_idx]
            this_t_mask = t_mask[batch_idx]
            this_prob = prob[this_s_mask][:, this_t_mask]
            this_sampled_s = diffusion_utils.metropolis_hastings_one_batch(this_prob, n_episodes)
            num_src_to_pad = len(this_s_mask) - len(this_sampled_s)
            if num_src_to_pad > 0:
                this_sampled_s = np.concatenate((this_sampled_s, np.zeros(num_src_to_pad)))
            sampled_s.append(this_sampled_s)
        sampled_s = torch.LongTensor(np.stack(sampled_s)).to(self.device)
        return sampled_s
    
    
    def perturb_sampling(self, p_probXs_givenXt, s_mask, t_mask):
        temperatures = torch.linspace(0.0001, 0.01, self.T)
        this_temperature = temperatures[int(t[0].item() * self.T) - 1]
        sampled_s = list()
        for batch_idx in range(bs):
            prob = p_probXs_givenXt[batch_idx].clone()
            this_s_mask = s_mask[batch_idx]
            this_t_mask = t_mask[batch_idx]
            this_sampled_s = diffusion_utils.perturb_sampling_one_batch(prob, this_temperature, self.device)[0]
            num_src_to_pad = len(this_s_mask) - len(this_sampled_s)
            if num_src_to_pad > 0:
                this_sampled_s = np.concatenate((this_sampled_s, np.zeros(num_src_to_pad)))
            sampled_s.append(this_sampled_s)
        sampled_s = torch.LongTensor(np.stack(sampled_s)).to(self.device)
        return sampled_s
    

    def on_validation_start(self):
        print('Start evaluation...')


    def validation_epoch_end(self, outs):
        accs = list()
        for out_ in outs: # for each data_loader output
            this_sum = 0
            this_acc = 0
            for out in out_: # for each mini_batch of dataloader
                acc, bs = out['test_acc'], out['bs']
                this_sum += bs 
                this_acc += acc * bs
            accs.append(this_acc / this_sum)
        acc = sum(accs) / len(accs) # average over all dataloders
        print("test/acc_epoch/mean: {:.4f}".format(acc))
        self.log("test/acc_epoch/mean", acc)
        print('Evaluation finished!')


    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()
        self.test_acc.reset()
        print("Start evaluating after {} epochs...".format(self.current_epoch))


    def on_test_epoch_start(self) -> None:
        print("Starting test...")
        
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
        print("test/acc_epoch/mean: ", acc)
        self.log("test/acc_epoch/mean", acc)
        print('Evaluation finished!')


    def get_transition_params(self, t_float, s_float, bs, num_classes, mask_transition):
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_sb = self.noise_schedule.get_alpha_bar(t_normalized=s_float) 
        alpha_tb = self.noise_schedule.get_alpha_bar(t_normalized=t_float) # this is okay!
        Qtb = self.transition_model.get_Qtb(alpha_tb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb = self.transition_model.get_Qtb(alpha_sb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qsb[torch.isnan(Qsb)] = 0.1
        Qtb[torch.isnan(Qtb)] = 0.1
        
        if isinstance(self.transition_model, MarginalUniMatchingTransition):
            Qt = self.transition_model.get_Qt(beta_t, alpha_sb, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        else:
            Qt = self.transition_model.get_Qt(beta_t, num_classes, batch_size=bs, device=self.device) * mask_transition  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        return {'Qsb': Qsb, 'Qtb': Qtb, 'Qt': Qt,'alpha_sb': alpha_sb, 'alpha_tb': alpha_tb}
        

    def configure_optimizers(self):
        # backbone_params = list(self.model.node_layers.parameters()) + list(self.model.edge_layers.parameters())
        # backbone_params += list(self.model.final_layers.parameters())

        # backbone_ids = [id(item) for item in backbone_params]

        # new_params = [param for param in self.model.parameters()] # if id(param) not in backbone_ids]
        # opt_params = [
        #     # dict(params=backbone_params, lr=self.cfg.train.lr * 0.01),
        #     dict(params=new_params, lr=self.cfg.train.lr)
        # ]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        lrc = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 50], gamma=0.4) # 0.002; 0.0007, 0.00025, 
        return [optimizer], [lrc]
    

    