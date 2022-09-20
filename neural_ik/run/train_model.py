import numpy as np
import tensorflow as tf

from neural_ik.models import converging_dnn, simple_dnn
from data.data_io import load_as_tf_dataset, load_dataset
from neural_ik.visual import plot_training_history
from neural_ik.loss_functions import DualQuatMormLoss
from data.robots import arm6dof
from datetime import datetime
from keras.losses import MSE
from keras.optimizers import Adagrad


def save_model(model, tag):
    model.save(f'./models/{model.name}___{tag}.hdf5')


def main():
    dataset_name = 'fk_ds_0_0_3'
    train_x, train_y, val_x, val_y = load_dataset(dataset_name)

    model = simple_dnn(train_x.shape[1], train_y.shape[1])

    loss = MSE  #DualQuatMormLoss(robot)
    opt = Adagrad()
    model.compile(optimizer=opt, loss=loss)
    tag = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    save_model(model, tag)

    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=100, batch_size=32)

    save_model(model, tag)
    plot_training_history(history, f'../pics/{dataset_name}_{model.name}___{tag}.png')


if __name__ == '__main__':
    main()
