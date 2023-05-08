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
        this_setting += '_greedy'
    return this_setting


@hydra.main(version_base='1.1', config_path='../configs', config_name='config_align')
def main(cfg: DictConfig):
    config = dict()

    # datamodule = pascalvoc.PascalVOCModule(cfg)
    # let define new dataloader!
    
    dataset_len = {"train": cfg.train.epoch_iters * cfg.train.batch_size, "test": cfg.train.eval_samples}
    image_dataset = {
        x: GMDataset(cfg.data_config.name, sets=x, length=dataset_len[x], obj_resize(256, 256)) for x in ("train", "test")
    }
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == "test")) for x in ("train", "test")}
    
    import pdb; pdb.set_trace()
    
    
    model_kwargs = {}

    if cfg.general.test_only:
        saved_cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    # update sampling cfg:
    saved_cfg.model.use_argmax = cfg.model.use_argmax
    saved_cfg.model.metropolis = cfg.model.metropolis
    saved_cfg.model.sample_mode = cfg.model.sample_mode
    cfg = saved_cfg

    #utils.create_folders(cfg)
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

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
                      enable_progress_bar=cfg.train.progress_bar,
                      logger=wandb_logger,
                      log_every_n_steps=cfg.train.log_every_n_steps,
                      fast_dev_run = cfg.general.name.lower() == 'debug',
                      callbacks=callbacks,
                      deterministic=False)


    VISUALIZE_SIZE=2
    VISUALIZE_RANDOM=False

    model.train_samples_to_visual = datamodule.visual_dataloader_train(shuffle=VISUALIZE_RANDOM, size=VISUALIZE_SIZE)
    model.test_samples_to_visual = datamodule.visual_dataloader_test(shuffle=VISUALIZE_RANDOM, size=VISUALIZE_SIZE)

    if not cfg.general.test_only:
        trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.pascal_test_, ckpt_path=cfg.general.resume) # ckpt_path is None
        trainer.test(model, dataloaders=datamodule.pascal_test_)
    else:
        trainer.test(model, dataloaders=datamodule.pascal_test_, ckpt_path=cfg.general.test_only)

    if cfg.general.remove_log:
        dir_path = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
        shutil.rmtree(dir_path)

if __name__ == '__main__':
    main()
