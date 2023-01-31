from train.trainer import Trainer
from train.dense_trainer import DenseTrainer

trainer_dict = {
    'trainer': Trainer,
    'densetrainer': DenseTrainer,
}

__all__ = ['trainer_dict']