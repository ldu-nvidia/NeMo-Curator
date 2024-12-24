### use this script to do SFT with NeMo2.0 framework
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
import pytorch_lightning as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed

from typing import List, Optional
from nemo.lightning.io.mixin import IOMixin
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy, 
        tensor_model_parallel_size = 8
    )
    trainer = run.Config (
        nl.Trainer,
        devices = 8,
        max_steps = 20,
        accelerator = "gpu",
        strategy = strategy,
        plugins = bf16_mixed(),
        log_every_n_steps = 1,
        limit_val_batches = 2,
        val_check_interval = 2, 
        num_sanity_val_steps = 0,
    )
    return trainer

def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last = True,
        every_n_train_steps = 10,
        monitor = "reduced_train_loss",
        save_top_k = 1,
        save_on_train_epoch_end = True,
        save_optim_on_train_end = True,
    )
    return run.Config(
        nl.NeMoLogger, 
        name = "nemo2_sft",
        log_dir = "./",
        use_datetime_version = True,
        ckpt = ckpt,
        wandb = None
    )

def adam_with_cosine_annealing() -> run.Config[nl.OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=5e-6,
        adam_beta2=0.98,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )
    return run.Config(
        nl.MegatronOptimizerModule,
        config=opt_cfg
    )

def llama3_8b() -> run.Config[pl.LightningModule]:
    return run.Config(llm.LlamaModel, config=run.Config(llm.Llama3Config8B))

def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig,
            path="nemo://meta-llama/Meta-Llama-3-8B"
        ),
        resume_if_exists=True,
    )

def configure_finetuning_recipe():
    return run.Partial(
        llm.finetune,
        model=llama3_8b(),
        trainer=trainer(),
        data=dolly(),
        log=logger(),
        optim=adam_with_cosine_annealing(),
        resume=resume(),
    )