import keras
import tensorflow as tf
from keras import Model
from keras import layers

from tf_kinematics.kinematic_models_io import load as load_kin
from tf_kinematics.layers.iso_layers import IsometryCompact, Diff
from tf_kinematics.layers.kin_layers import ForwardKinematics
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.sys_solve import iter_grad

from neural_ik.losses import qp_loss, QP


class QPRecurrentGradBoost(tf.keras.Model):
    def __init__(self, kin_model_name: str, batch_size: int, n_iters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel = load_kin(kin_model_name, batch_size)
        dof = kernel.dof
        activation = tf.nn.relu

        self.n_iters = n_iters
        self.iso_compact = IsometryCompact()

        self.target_fn = tf.function(lambda theta: tf_compact(kernel.forward(tf.reshape(theta, [-1]))))

        self.lr_corrector = keras.Sequential([
            layers.Dense(16, activation=activation),
            layers.Dense(32, activation=activation),
            layers.Dense(dof, activation='sigmoid')
        ], name='gradient_boost')

        # self.beta_corrector = keras.Sequential([
        #     layers.Dense(16, activation=activation),
        #     layers.Dense(32, activation=activation),
        #     layers.Dense(dof, activation='sigmoid')
        # ], name='beta_boost')

        # y_shape = 6
        # self.q_corrector = keras.Sequential([
        #     layers.Dense(16, activation=activation),
        #     layers.Dense(32, activation=activation),
        #     layers.Dense(y_shape, activation='tanh'),
        # ], name='beta_boost')
        #
        # self.p_corrector = keras.Sequential([
        #     layers.Dense(16, activation=activation),
        #     layers.Dense(32, activation=activation),
        #     layers.Dense(y_shape * y_shape, activation='tanh'),
        #     layers.Reshape(target_shape=(y_shape, y_shape))
        # ], name='beta_boost')

        self.fk_iso = ForwardKinematics(kin_model_name, batch_size)
        self.fk_compact = IsometryCompact()
        self.fk_diff = Diff()

    def call(self, inputs: (tf.Tensor, tf.Tensor), training=None, mask=None):
        theta_seed, iso_goal = inputs
        gamma = self.iso_compact(iso_goal)

        theta = theta_seed
        # momentum = tf.zeros_like(theta)
        batch_size = theta_seed.shape[0]
        q = tf.zeros(shape=(batch_size, 6))
        p = tf.eye(6, batch_shape=[batch_size])
        for _ in range(self.n_iters):
            loss = tf.losses.mean_squared_error #lambda y_g, y: qp_loss(tf.abs(y - y_g), p, q)

            grad = iter_grad(gamma, theta, self.target_fn, loss)
            grad_gamma_and_seed = tf.concat([grad, gamma, theta], axis=-1)

            lr = self.lr_corrector(grad_gamma_and_seed)
            # beta = self.beta_corrector(grad_gamma_and_seed)

            # lr = tf.clip_by_value(lr, clip_value_min=0.0001, clip_value_max=0.9999)
            # beta = tf.clip_by_value(beta, clip_value_min=0.0001, clip_value_max=0.9999)
            # momentum = (1.0 - beta) * momentum + beta * grad

            theta = theta - lr * grad

            # gamma_theta = tf.concat([gamma, theta], axis=-1)
            # p += self.p_corrector(gamma_theta)
            # q += self.q_corrector(gamma_theta)

        fk = self.fk_iso(theta)
        fk_compact = self.fk_compact(fk)
        ft_diff = self.fk_diff([fk_compact, gamma])
        return ft_diff


def qp_recurrent_grad_boost(kin_model_name: str, batch_size: int, n_iters: int) -> Model:
    assert n_iters > 0
    model = QPRecurrentGradBoost(kin_model_name, batch_size, n_iters, name='qp_recurrent_grad_boost')

    dof = load_kin(kin_model_name, batch_size).dof

    theta = tf.ones(shape=(batch_size, dof))
    goal = tf.ones(shape=(batch_size, 4, 4))
    model.call([theta, goal])
    model.build([(batch_size, dof), (batch_size, 4, 4)])
    return model
