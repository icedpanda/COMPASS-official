import json
import time

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig

from src.llmcrs.data import RedialDataset, ReDialDataModule
from src.llmcrs.data.datatype import TaskType
from src.llmcrs.engines import BaseSystem
from src.llmcrs.models.lightning_modules import COMPASSLightning
from src.llmcrs.utils import log_hyperparameters, log_model_parameters


class COMPASSSystem(BaseSystem):
    def __init__(self, dataset: RedialDataset, config: DictConfig):
        super().__init__(dataset, config)

    def init_models(self,
                    phase: TaskType = TaskType.REC,
                    warmup_steps: int = None,
                    total_steps: int = None,
                    best_rec_model_path: str = None,
                    ) -> LightningModule:
        """
        Initialize callbacks.
        """
        edge = self.dataset.kg["edge"]
        n_entity = len(self.dataset.kg["entity2id"]) + 1
        n_relation = len(self.dataset.kg["id2relation"])
        net_config = {
            "edges": list(edge),
            "n_ent": n_entity,
            "n_rel": n_relation,
        }
        config = self._get_config_for_phase(phase, "model_config").pl
        logger.info(f"Initializing models for {phase} phase... {config._target_}")
        model: COMPASSLightning = instantiate(
            config,
            net=net_config,
            item_entity_ids=self.labels_list,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            phase=phase,
            id2entity=self.dataset.kg["id2entity"],
            is_ddp=self.is_ddp,
        )
        logger.info("Model initialized successfully!")
        logger.info(f"Phase: {phase} Model path: {best_rec_model_path}")
        if best_rec_model_path:
            logger.info(f"INTI MODEL: Loading best model from {best_rec_model_path}")
            ckpt = torch.load(best_rec_model_path)
            model.load_state_dict(ckpt["state_dict"])

        return model

    def init_dataloader(self, phase: TaskType = TaskType.REC) -> ReDialDataModule:
        dataloader_config = self._get_config_for_phase(phase, 'dataloader_config')
        logger.info(f"Initializing dataloader <{dataloader_config._target_}>")
        dataloader: ReDialDataModule = instantiate(
            dataloader_config,
            dataset=self.dataset,
            phase=phase,
        )
        return dataloader

    def train_phase(self, phase: TaskType, best_model_path: str = None):
        logger.info(f"{phase.value} training started")
        dataloader = self.init_dataloader(phase)
        callbacks = self.init_callbacks(phase)
        use_warmup = self.is_warmup_scheduler(self.model_config[phase.value].pl.scheduler)
        if use_warmup:
            total_steps, warm_up_steps = self.log_steps(dataloader, phase)
        else:
            total_steps, warm_up_steps = None, None

        model = self.init_models(phase, warmup_steps=warm_up_steps, total_steps=total_steps,
                                 best_rec_model_path=best_model_path)
        # Initialize the trainer
        trainer = self.init_trainer(phase, callbacks=callbacks, loggers=self.loggers)

        # log_everything
        log_hyperparameters(self.config, trainer)
        log_model_parameters(model, trainer, phase=phase)
        logger.info(f"Start {phase.value} training")
        # model training
        trainer.fit(model, dataloader)
        logger.info(f"{phase.value} training finished")
        logger.info("start testing !")
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        logger.info(f"best model path: {best_checkpoint}")
        # test inference time
        logger.warning("Start testing inference time")
        t0 = time.time()
        trainer.test(model, dataloader)
        t1 = time.time()
        logger.warning(f"Test inference time: {t1 - t0}")

        return best_checkpoint

    def fit(self):
        best_checkpoint = None
        if self.pre_train:
            best_checkpoint = self.train_pre()
        logger.warning(f"Load best model from pre-training phase: {best_checkpoint}")
        self.train_rec(best_checkpoint)
        self.clean_files()

    def save_outputs(self, predictions, loader_type: str = "test"):
        # for each batch, it will have
        # 1. llm_inputs, list of strings
        # 2.generated_text, list of strings
        # 3. llm_outs, list of strings
        # 4. context_entities, list of list of strings
        # 5. outputs.last_hidden_states, bs, dim
        # 6. utt_ids
        # save all batch outputs to a json file
        # use utt_ids as key
        json_file_path = f"{self.paths.output_dir}/{loader_type}_generated_outputs.json"
        pt_file_path = f"{self.paths.output_dir}/{loader_type}_last_hidden_states.pth"
        logger.info(f"Saving {loader_type} outputs to {json_file_path}")
        outputs_to_save = {}
        tensor_data = {}
        for batch in predictions:
            llm_inputs, generated_text, llm_outs, context_entities, last_hidden_states, utt_ids = batch
            for i, utt_ids in enumerate(utt_ids):
                outputs_to_save[utt_ids] = {
                    "llm_inputs": llm_inputs[i],
                    "generated_text": generated_text[i],
                    "llm_outs": llm_outs[i],
                    "context_entities": context_entities[i],
                }
                tensor_data[utt_ids] = last_hidden_states[i]

        with open(json_file_path, "w") as f:
            json.dump(outputs_to_save, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {loader_type} outputs to {json_file_path}")

        torch.save(tensor_data, pt_file_path)
        logger.info(f"Saved {loader_type} last hidden state outputs to {pt_file_path}")
