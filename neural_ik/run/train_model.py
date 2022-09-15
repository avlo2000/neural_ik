import numpy as np
import tensorflow as tf

from neural_ik.models import converging_dnn, simple_dnn
from data.generators import TrjGen, RandomGen
from neural_ik.visual import plot_training_history
from neural_ik.loss_functions import DualQuatMormLoss
from data.robots import arm6dof
from datetime import datetime
from keras.metrics import MeanSquaredError

Generator = RandomGen


def save_model(model, tag):
    model.save(f'./models/{model.name}___{tag}.hdf5')


def main():
    robot = arm6dof()
    dof = robot.num_axis
    ws_lim = np.zeros((dof, 2), dtype=np.float32)
    ws_lim[:, 1] = [np.pi] * dof
    ws_lim[:, 0] = [-np.pi] * dof

    gen_train = Generator(batch_size=32, robot=robot, n=100)
    gen_valid = Generator(batch_size=32, robot=robot, n=100)
    model = simple_dnn(6, gen_train.output_dim)

    opt = tf.keras.optimizers.RMSprop()
    loss = MeanSquaredError() #DualQuatMormLoss(robot)
    model.compile(optimizer=opt, loss=loss)

    history = model.fit_generator(generator=gen_train, validation_data=gen_valid, epochs=6)

    tag = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    plot_training_history(history, f'../pics/{Generator.__name__}_{model.name}___{tag}.png')
    save_model(model, tag)


if __name__ == '__main__':
    main()
