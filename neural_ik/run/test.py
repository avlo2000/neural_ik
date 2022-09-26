import tensorflow as tf
from keras.losses import MeanSquaredError
from tqdm import tqdm

from neural_ik.models import residual_fk_dnn


from tf_kinematics.kinematic_models import kuka_robot


if __name__ == '__main__':
    batch_size = 1
    kin = kuka_robot(batch_size)

    model, thera_out = residual_fk_dnn(kin)
    model.summary()

    opt = tf.keras.optimizers.RMSprop()
    loss = MeanSquaredError()
    model.compile(optimizer=opt, loss=loss)

    n = 500
    thetas_rnd_seed = []
    gammas = []

    for _ in tqdm(range(n)):
        theta_rnd_seed = tf.random.uniform(shape=(1, kin.dof))
        theta_rnd = tf.random.uniform(shape=(1, kin.dof))
        gamma = kin.forward(tf.reshape(theta_rnd, [-1]))

        thetas_rnd_seed.append(theta_rnd_seed)
        gammas.append(gamma)
    thetas_rnd_seed = tf.squeeze(tf.stack(thetas_rnd_seed))
    gammas = tf.squeeze(tf.stack(gammas))

    y = tf.squeeze(tf.stack([tf.eye(4)] * n))

    model.fit(x=[thetas_rnd_seed, gammas], y=y, epochs=10, batch_size=batch_size)


