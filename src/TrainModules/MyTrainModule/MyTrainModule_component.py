from dneg_ml_toolkit.src.Component.component_store import ComponentStore
from dneg_ml_toolkit.src.TrainModules.BASE_TrainModule.BASE_TrainModule_component import BASE_TrainModule

from src.TrainModules.MyTrainModule.MyTrainModule_config import MyTrainModuleConfig
from dneg_ml_toolkit.src.Losses.BASE_Loss.BASE_Loss_component import BASE_Loss
from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.DataModules.DataModule.DataModule_component import DataModule
from dneg_ml_toolkit.src.Data.Dataloaders.Dataloader.Dataloader_component import Dataloader

from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torchmetrics import Accuracy, MeanMetric

from typing import List, Dict, Any, cast


class MyTrainModule(BASE_TrainModule):
    """

    Args:
        config: ComponentConfig object for the TrainModule
        experiment_name: Name of the experiment
        experiment_folder: Save folder for the experiment
    """

    def __init__(self, config: MyTrainModuleConfig, experiment_name: str, experiment_folder: str):

        super().__init__(config=config, experiment_name=experiment_name, experiment_folder=experiment_folder)
        self.config: MyTrainModuleConfig = config

        self.Network: BASE_Network = cast(BASE_Network,
                                          ComponentStore().build_component_from_config(self.config.Network,
                                                                                       input_shape=self.config.InputShape))
        self.Losses: List[BASE_Loss]
        if isinstance(self.config.Loss, list):
            # Support multiple losses
            self.Losses = [cast(BASE_Loss, ComponentStore().build_component_from_config(loss)) for loss in
                           self.config.Loss]
        else:
            # If only a single loss is configured, create a list with one element, since the training loop
            # will loop over a list of losses
            self.Losses = [cast(BASE_Loss, ComponentStore().build_component_from_config(self.config.Loss))]

        self.Optimizer = ComponentStore().build_component_from_config(self.config.Optimizer,
                                                                      optimizer_params=self.get_training_parameters())

        self.Scheduler = None
        if self.config.Scheduler is not None:
            self.Scheduler = ComponentStore().build_component_from_config(self.config.Scheduler)

        self.train_metrics = {
            'accuracy': Accuracy(task='binary'),
            'loss': MeanMetric()
        }

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        """
        Lightning function to inform Lightning what optimizers and LR schedulers are being used

        Returns:
            List of configuration for each optimizer
        """
        optimizers = []
        optimizer_configuration = {"optimizer": self.Optimizer}

        # If a scheduler has been built, add it to the configuration for Lightning
        if self.Scheduler is not None:
            optimizer_configuration["lr_scheduler"] = {"scheduler": self.Scheduler,
                                                       "interval": self.config.Scheduler.Interval.value,
                                                       # Get the interval from the config's enum
                                                       "name": "lr"
                                                       }

        optimizers.append(optimizer_configuration)

        return optimizers

    def get_training_parameters(self) -> List[torch.nn.parameter.Parameter]:
        """
        Get the trainable parameters for the Network

        Returns:
            List of training parameters
        """

        parameters = []
        parameters.extend(list(self.Network.parameters()))

        return parameters

    def forward(self, batch_data: MLToolkitDictionary) -> MLToolkitDictionary:
        """
        Perform a forward pass on the Network

        Args:
            batch_data: The Data for the batch, which gets passed into the Network

        Returns:
            The network output

        """

        return self.Network.forward(batch_data, step=self._train_step)

    def training_step(self, batch, batch_idx):
        """
        Implements the training loop for the model.

        Args:
            batch: Current data batch
            batch_idx:

        Returns:

        """

        step_metrics = {}

        data, metadata = batch
        # Ensure that the created tensor is on the same device as the data
        device = data['data'].device
        total_loss = torch.zeros(1)
        total_loss = total_loss.to(device)

        network_outputs = self.forward(data)

        #batch_accuracy = self.train_metrics['accuracy'](
            #network_outputs["data"].detach().cpu().softmax(-1), metadata["target"].cpu())

        #step_metrics["batch_accuracy"] = batch_accuracy

        for loss_function in self.Losses:
            loss = loss_function(network_outputs, metadata)
            total_loss += loss

            loss_name = loss_function.Name()
            step_metrics["step/{}".format(loss_name)] = loss.item()

        #self.train_metrics['loss'].update(total_loss.detach().cpu())

        step_metrics["batch_loss"] = total_loss.item()

        # # Log the total loss separately with prog_bar=True so it will show in the progress bar
        # self.log("loss", total_loss.item(), on_step=True, on_epoch=True, logger=True, prog_bar=True)

        # Log the step metrics
        self.log_dict(step_metrics, on_step=True, logger=True)

        return total_loss

    def training_epoch_end(self, outputs):
        #self._log_epoch_metrics(self.train_metrics, prefix='train/')
        pass

    def validation_step(self, batch, batch_idx):
        """
        Called during training to perform validation
        Args:
            batch:
            batch_idx:

        Returns:

        """

        data, targets = batch
        original_data = data['data'].clone().detach()
        network_outputs = self.forward(data)

        imagesToLog = []
        for idxToLog in range(data['data'].shape[0]):
            source = original_data[idxToLog,:,:,:]

            target = targets['target'][idxToLog,:,:,:]
            target = torch.cat([target, target, target], 0)

            output = network_outputs['data'][idxToLog,:,:,:]
            output = torch.cat([output, output, output], 0)

            imagesToLog.append(torch.cat([source, target, output], 1))


        imageToLog = torch.cat(imagesToLog,2)

        #self.val_metrics['accuracy'](
            #network_outputs["data"].detach().cpu().softmax(-1), targets["target"].cpu())

        #device = data['data'].device
        #total_loss = torch.zeros(1)
        #total_loss = total_loss.to(device)

        #for loss_function in self.Losses:
            #loss = loss_function(network_outputs, targets)
            #total_loss += loss

            #loss_name = loss_function.Name()

        #self.val_metrics['loss'].update(total_loss.detach().cpu())


        #image_name = "validation_id_{}".format(id)

        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):  # Get the tensorboard logger
                tb_logger = logger.experiment

                tb_logger.add_image('validation', imageToLog, global_step=self.global_step)
                tb_logger.flush()


    def _log_epoch_metrics(self, metrics, prefix='train/'):
        logs = {}
        for key, metric in metrics.items():
            logs["{}{}".format(prefix, key)] = metric.compute()
            metric.reset()

        self.log_dict(logs, on_epoch=True, logger=True)
