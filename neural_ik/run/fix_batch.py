from __future__ import absolute_import

import tensorflow as tf
from keras import layers

from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.kinematic_models_io import load
from tf_kinematics.urdf_parser.urdf import Robot  # noqa
from tf_kinematics.layers.kin_layers import ForwardKinematics as FK
from tf_kinematics.layers.solve_layers import SolveIterGrad, SolveCompactIterGrad


class ForwardKinematics(layers.Layer):
    def __init__(self, urdf_file, base_link, end_link, batch_size, **kwargs):
        self.urdf_file = urdf_file
        self.base_link = base_link
        self.end_link = end_link
        self.batch_size = batch_size
        super(ForwardKinematics, self).__init__(**kwargs)

    def build(self, input_shape):
        dlkinematics_chain = chain_from_urdf_file(self.urdf_file)
        self.kernel = DLKinematics(
            dlkinematics_chain, self.base_link, self.end_link, self.batch_size)

    def call(self, input):
        return self.kernel.forward(tf.reshape(input, [-1]))

    def compute_output_shape(self, input_shape):
        return (self.batch_size, 4, 4)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'urdf_file': self.urdf_file,
            'base_link': self.base_link,
            'end_link': self.end_link,
            'batch_size': self.batch_size,
        })
        return config


from tf_kinematics.urdf import chain_from_urdf_file
import keras


# FK_layer = ForwardKinematics(
#     urdf_file=r'C:\Users\Pavlo\source\repos\math\neural_ik\urdf\kuka.urdf',
#     base_link='calib_kuka_arm_base_link',
#     end_link='kuka_arm_7_link',
#     batch_size=10)

# FK_layer = FK(kin_model_name='kuka_robot', batch_size=10)
FK_layer = SolveIterGrad(loss_ident='mse', kin_model_name='kuka_robot', batch_size=10)

# model = keras.Sequential()
# model.add(layers.Dense(7))
# model.add(FK_layer)
# model.add(layers.Flatten())
# model.add(layers.Dense(10))

inp = layers.Input(7)
x = layers.Dense(7)(inp)
x = SolveCompactIterGrad(loss_ident='mse', kin_model_name='kuka_robot', batch_size=10)([tf.ones(shape=(10, 6)), x])
x = layers.Flatten()(x)
x = layers.Dense(10)(x)

model = keras.Model(inputs=inp, outputs=x)
model.compile(loss='mse')

x = tf.ones(shape=(100, 7))
y = tf.ones(shape=(100, 10))
model.fit(x, y, batch_size=10, validation_split=0.13, validation_batch_size=10)
