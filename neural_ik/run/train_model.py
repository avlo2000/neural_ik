import numpy as np
import tensorflow as tf

from neural_ik.models import converging_dnn
from neural_ik.generators import TrjGen
from neural_ik.visual import plot_training_history
from neural_ik.robots import arm6dof
from datetime import datetime


Generator = TrjGen


def save_model(model, tag):
    model.save(f'./models/{model.name}___{tag}.hdf5')


def main():
    robot = arm6dof()
    dof = robot.num_axis
    ws_lim = np.zeros((dof, 2), dtype=np.float32)
    ws_lim[:, 1] = [np.pi] * dof
    ws_lim[:, 0] = [-np.pi] * dof

    gen_train = Generator(ws_lim=ws_lim, robot=robot, trj_size=100, trj_count=1, step=0.02)
    gen_valid = Generator(ws_lim=ws_lim, robot=robot, trj_size=100, trj_count=1, step=0.02)
    model = converging_dnn(6, gen_train.output_dim)

    opt = tf.keras.optimizers.RMSprop()
    loss = tf.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss)

    history = model.fit_generator(generator=gen_train, validation_data=gen_valid, epochs=6)

    tag = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    plot_training_history(history, f'../pics/{Generator.__name__}_{model.name}___{tag}.png')
    save_model(model, tag)


if __name__ == '__main__':
    main()
