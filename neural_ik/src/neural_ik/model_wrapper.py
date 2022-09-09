from datetime import datetime
from typing import Callable

from neural_ik.abstract_generator import FKGenerator
from neural_ik.visual import plot_training_history

import tensorflow as tf


class ModelWrapper:
    def __init__(self, generator: FKGenerator):
        self.model = None
        self.generator = generator

    def train_and_save(self, model: tf.keras.Model, train_func: Callable[[tf.keras.Model, FKGenerator],
                                                                         tf.keras.History]):
        self.model = model

        tag = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
        history = train_func(self.model, self.generator)
        plot_training_history(history, save_path=f'./pics/{type(self.generator).__name__}:{model.name}_{tag}.png')
        model.save(f'./models/{model.name}_{tag}.hdf5')

    def load_and_evaluate(self, path_to_model: str):
        model = tf.keras.models.load_model(path_to_model)
        loss = model.evaluate_generator(self.generator, verbose=1)
        print(f"Loss: {loss}")
