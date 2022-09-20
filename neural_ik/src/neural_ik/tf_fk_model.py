import tensorflow as tf
from visual_kinematics import RobotSerial


class TFFKModel(tf.keras.layers.Layer):
    def __int__(self, robot: RobotSerial):
        super(TFFKModel, self).__init__()
        self.__robot = robot

    def build(self, input_shape):
        self.add_variable
