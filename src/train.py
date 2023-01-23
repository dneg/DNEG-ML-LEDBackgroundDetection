import os
from typing import cast, Optional

from dneg_ml_toolkit.src.Train_config import TrainConfig
from dneg_ml_toolkit.src.Component.component_store import ComponentStore
from dneg_ml_toolkit.src.Data.DataModules.DataModule.DataModule_component import DataModule
from dneg_ml_toolkit.src.utils import device_utils
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit.src.checkpoints import checkpoint_utils
from dneg_ml_toolkit.src.utils.logger import Logger

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


def train(training_config: TrainConfig, resume: bool, resume_checkpoint: Optional[str] = None) -> None:
    """
    Run the Pytorch Lightning Training for the project using the provided configuration object
    Args:
        training_config: A config object created by loading the configuration file for the experiment
        resume: Flag, if true will attempt to load the latest saved checkpoint for the experiment
        resume_checkpoint: Path to a specific checkpoint to resume from
    Returns:
        None
    """

    # 1. Build the Lightning DataModule from the configuration
    Logger().Log("--------------------Building Data Module----------")
    data_module = cast(DataModule, ComponentStore().build_component_from_config(training_config.DataModule))
    Logger().Log("--------------------Data Module Built----------")

    # 2. Build the Lightning Training Module from the configuration
    Logger().Log("--------------------Building Train Module----------")
    # Resolve the train_module config to the corresponding Component class registered in the component store
    train_module = cast(LightningModule,
                        ComponentStore().build_component_from_config(training_config.TrainModule,
                                                                     experiment_name=training_config.Name,
                                                                     experiment_folder=training_config.Experiment_Folder,
                                                                     data_module=data_module))
    Logger().Log("--------------------Train Module Built----------")

    device_config = device_utils.get_lightning_device_configuration(device=training_config.Device,
                                                                    multi_gpu_strategy="ddp")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_folder = os.path.join(training_config.Experiment_Folder, Globals().CHECKPOINTS_FOLDER)

    # TODO
    # checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_folder, monitor='val/loss', mode='min',
    #                                       filename="{}".format(training_config.Name) + "_{epoch:03d}"
    #                                       )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        every_n_epochs=1,
        save_top_k=-1,  # Save all checkpoints
        filename="{}".format(training_config.Name) + "_{epoch:03d}",
    )

    report_folder = os.path.join(training_config.Experiment_Folder, Globals().REPORTS_FOLDER)
    tensorboard_logger = TensorBoardLogger(save_dir=report_folder, name=training_config.Name)

    if resume:
        resume_checkpoint = checkpoint_utils.get_latest_checkpoint(checkpoint_folder)

    Logger().Log("--------------------Building Lightning Trainer----------")
    trainer = Trainer(max_epochs=training_config.Epochs,
                      callbacks=[checkpoint_callback, lr_monitor],
                      resume_from_checkpoint=resume_checkpoint,
                      logger=tensorboard_logger,
                      log_every_n_steps=training_config.LogInterval,
                      # val_check_interval=1.0,
                      check_val_every_n_epoch=1,
                      limit_val_batches=0,  # Run validation on a single batch
                      num_sanity_val_steps=0,  # Run 1 batch of validation for sanity checking
                      **device_config)
    Logger().Log("--------------------Lightning Trainer Built----------")

    Logger().Log("--------------------Starting Training----------")
    trainer.fit(train_module, data_module)
    Logger().Log("--------------------Training Complete----------")
