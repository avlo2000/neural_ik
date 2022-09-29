from pathlib import Path

import tensorflow as tf

from data.tf_kin_data import rawdata_to_dataset
from neural_ik.losses import ExpWeightedMSE, exp_weighted_mse
from data.data_io import load_as_tf_dataset, read_csv

from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.models.residual_newton_iter_dnn import residual_newton_iter_dnn


from tf_kinematics.kinematic_models import kuka_robot


PATH_TO_DATA = Path('../data').absolute()
PATH_TO_MODELS = Path('../models').absolute()

if __name__ == '__main__':
    batch_size = 16
    kin = kuka_robot(batch_size)

    blocks_count = 30
    model_dist, model_ik = residual_fk_dnn(kin, blocks_count=blocks_count, corrector_units=20)
    model_dist.summary()

    opt = tf.keras.optimizers.RMSprop()
    loss = ExpWeightedMSE()
    model_dist.compile(optimizer=opt, loss=loss)

    with open(PATH_TO_DATA/'kuka_train_10k.csv', mode='r') as file:
        feature_names, raw_data = read_csv(file)
    thetas, thetas_seed, iso_transforms = rawdata_to_dataset(kin, feature_names, raw_data)
    # y = tf.squeeze(tf.stack([tf.eye(4)] * n))
    y = tf.zeros(shape=(len(thetas), blocks_count), dtype=float)

    model_dist.fit(x=[thetas_seed, iso_transforms], y=y, epochs=10, batch_size=batch_size, validation_split=0.1)
    model_dist.save(PATH_TO_MODELS/f'{model_dist.name}_0.hdf5')
    model_ik.save(PATH_TO_MODELS/f'{model_ik.name}_0.hdf5')
