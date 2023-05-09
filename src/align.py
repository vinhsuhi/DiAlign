# These imports are tricky because they use c++, do not move them
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import shutil
import pathlib
import warnings

import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from src.datasets.data_loader_multigraph import GMDataset, get_dataloader

from src import utils
from datasets import pascalvoc
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from copy import deepcopy

seed_everything(42, workers=True)
warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)

    saved_cfg = model.cfg
    saved_cfg.general.test_only = resume
    saved_cfg.general.name = name
    return saved_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def get_setting(cfg):
    this_setting = "t{}_T{}_{}_{}_{}".format(cfg.model.use_time, cfg.model.diffusion_steps, cfg.model.loss_type, cfg.model.sample_mode, cfg.train.lr)
    if cfg.model.metropolis:
        this_setting = this_setting + '_metro'
    if cfg.model.loss_type == 'hybrid' or cfg.model.loss_tpye == 'lvb_advance':
        this_setting += '_ce{:.4f}_vb{:.4f}'.format(cfg.model.ce_weight, cfg.model.vb_weight)  
    if cfg.model.use_argmax:
        this_setting += '_greedy_new'
    return this_setting


@hydra.main(version_base='1.1', config_path='../configs', config_name='config_align')
def main(cfg: DictConfig):
    config = dict()

    # datamodule = pascalvoc.PascalVOCModule(cfg)
    # let define new dataloader!
    
    dataset_len = {"train": cfg.train.epoch_iters * cfg.train.batch_size, "test": cfg.train.eval_samples}
    
    image_dataset_train = GMDataset(cfg.dataset.name, sets='train', cfg=cfg, length=dataset_len['train'], obj_resize=(256, 256), base_dir=cfg.general.base_dir, exclude_willow_classes=cfg.dataset.exclude_willow_classes)
    batch_sizes = {'train': cfg.train.batch_size, 'test': cfg.train.batch_size_test}
    dataloader_train = get_dataloader(image_dataset_train, fix_seed=False, batch_size=batch_sizes['train'])
    # dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == "test"), batch_size=batch_sizes[x]) for x in ("train", "test")}
    dataloader_train.dataset.set_num_graphs(2)
    classes = dataloader_train.dataset.classes
    data_tests = list()
    data_vals = list()
    for cls in classes:
        im_data_test = GMDataset(cfg.dataset.name, sets='test', cfg=cfg, length=dataset_len['test'], obj_resize=(256, 256), base_dir=cfg.general.base_dir, exclude_willow_classes=cfg.dataset.exclude_willow_classes)
        im_data_val = GMDataset(cfg.dataset.name, sets='test', cfg=cfg, length=dataset_len['test'], obj_resize=(256, 256), base_dir=cfg.general.base_dir, exclude_willow_classes=cfg.dataset.exclude_willow_classes)
        data_tests.append(get_dataloader(im_data_test, fix_seed=True, batch_size=batch_sizes['test']))
        data_tests[-1].dataset.set_num_graphs(2)
        data_tests[-1].dataset.set_cls(cls)
        
        data_vals.append(get_dataloader(im_data_val, fix_seed=True, batch_size=batch_sizes['test']))
        data_vals[-1].dataset.set_num_graphs(2)
        data_vals[-1].dataset.set_cls(cls)
    
    model_kwargs = {}

    if cfg.general.test_only:
        saved_cfg, _ = get_resume(cfg, model_kwargs)
        # update sampling cfg:
        saved_cfg.model.use_argmax = cfg.model.use_argmax
        saved_cfg.model.metropolis = cfg.model.metropolis
        saved_cfg.model.sample_mode = cfg.model.sample_mode
        cfg = saved_cfg
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    

    #utils.create_folders(cfg)
    model = DiscreteDenoisingDiffusion(cfg=cfg, val_names=classes, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='test/acc_epoch/mean',
                                              save_top_k=3,
                                              mode='max',
                                              every_n_epochs=cfg.general.check_val_every_n_epochs)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=2)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    # library bug!
    # progress_bar = RichProgressBar(
    #     theme=RichProgressBarTheme(
    #         description="green_yellow",
    #         progress_bar="green1",
    #         progress_bar_finished="green1",
    #         progress_bar_pulse="#6206E0",
    #         batch_progress="green_yellow",
    #         time="grey82",
    #         processing_speed="grey82",
    #         metrics="grey82",
    #     )
    # )
    # callbacks.append(progress_bar)

    name = cfg.general.name
    if name == 'test':
        print("[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    this_setting = get_setting(cfg)    
    print(this_setting)
    
    
    wandb_logger = WandbLogger(project=f'dialign_{cfg.dataset.name}', name=this_setting)

    # we can keep trainer
    trainer = Trainer(
                      gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=cfg.train.progress_bar, logger=wandb_logger,
                      log_every_n_steps=cfg.train.log_every_n_steps,
                      fast_dev_run = cfg.general.name.lower() == 'debug',
                      callbacks=callbacks,
                      deterministic=False)


    #VISUALIZE_SIZE=2
    #VISUALIZE_RANDOM=False

    #model.train_samples_to_visual = datamodule.visual_dataloader_train(shuffle=VISUALIZE_RANDOM, size=VISUALIZE_SIZE)
    #model.test_samples_to_visual = datamodule.visual_dataloader_test(shuffle=VISUALIZE_RANDOM, size=VISUALIZE_SIZE)

    if not cfg.general.test_only:
        trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=data_vals, ckpt_path=cfg.general.resume) # ckpt_path is None
        trainer.test(model, dataloaders=data_tests)
    else:
        trainer.test(model, dataloaders=data_tests, ckpt_path=cfg.general.test_only)

    if cfg.general.remove_log:
        dir_path = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
        shutil.rmtree(dir_path)

if __name__ == '__main__':
    main()
