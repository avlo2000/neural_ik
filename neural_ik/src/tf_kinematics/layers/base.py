from keras.layers import Layer
import tensorflow as tf

from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.kinematic_models_io import load


@tf.keras.utils.register_keras_serializable()
class _KinematicLayer(Layer):
    def __init__(self, kin_model_name: str, batch_size: int, **kwargs):
        self._kernel: DLKinematics = load(kin_model_name, batch_size)
        self.__kin_model_name = kin_model_name
        self.__batch_size = batch_size
        super(_KinematicLayer, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config: dict):
        return cls(config['kin_model_name'], config['batch_size'])

    def get_config(self):
        config = super(_KinematicLayer, self).get_config()
        config.update({'kin_model_name': self.__kin_model_name})
        config.update({'batch_size': self.__batch_size})
        return config