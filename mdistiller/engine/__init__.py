from .trainer import BaseTrainer, CRDTrainer, KD_Trainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "tech": KD_Trainer,
}
