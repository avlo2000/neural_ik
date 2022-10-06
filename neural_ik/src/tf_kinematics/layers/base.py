from keras.layers import Layer
import tensorflow as tf

from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.kinematic_models_io import load


@tf.keras.utils.register_keras_serializable()
class _KinematicLayer(Layer):
    def __init__(self, kin_model_name: str, batch_size: int, *args, **kwargs):
        self._kernel = None
        self.__kin_model_name = kin_model_name
        self.__batch_size = batch_size
        super(_KinematicLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._kernel: DLKinematics = load(self.__kin_model_name, self.__batch_size)

    def get_config(self):
        config = super(_KinematicLayer, self).get_config()
        config.update({'kin_model_name': self.__kin_model_name})
        config.update({'batch_size': self.__batch_size})
        return config
