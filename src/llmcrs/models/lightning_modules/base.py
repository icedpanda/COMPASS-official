from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseLightningModule(LightningModule):
    def configure_optimizers(self):
        """Method to configure optimizers and schedulers.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configuration.
            The keys are 'optimizer' and 'lr_scheduler'.
        """
        optimizer = self.hparams.optimizer(params=[p for p in self.parameters() if p.requires_grad])

        if self.hparams.scheduler is not None:
            scheduler, scheduler_config = self._create_scheduler_and_config(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        return {"optimizer": optimizer}

    def _create_scheduler_and_config(self, optimizer):
        """Create the learning rate scheduler and its configuration"""
        scheduler = self._create_scheduler(optimizer)

        # Set default scheduler configuration
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "name": f"lr/{self.phase}/{optimizer.__class__.__name__}",
            "frequency": 1,
        }

        if isinstance(scheduler, ReduceLROnPlateau):
            # Adjust configuration for ReduceLROnPlateau scheduler
            scheduler_config["interval"] = "epoch"
            scheduler_config["monitor"] = f"val/{self.phase}/epoch_loss"

        return scheduler, scheduler_config

    def _create_scheduler(self, optimizer):
        """Create the learning rate scheduler"""
        if self.hparams.total_steps:
            # create the scheduler with warmup from transformers
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps,
            )
            self.print(f"Using transformer scheduler: {scheduler.__class__.__name__}")
        else:
            # create the scheduler without a warmup
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            self.print(f"Using scheduler: {scheduler.__class__.__name__}")

        return scheduler
