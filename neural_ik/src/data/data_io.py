import os
import csv

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from typing import Tuple, Iterable, Any, TextIO

from typing.io import IO
from visual_kinematics import Frame
from data.abstract_generator import FKGenerator


def zip_as_tf_dataset(x, y) -> tf.data.Dataset:
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)
    return tf.data.Dataset.zip((dx, dy))


def load_as_tf_dataset(dataset_name: str) -> (tf.data.Dataset, tf.data.Dataset):
    x_train, y_train, x_test, y_test = load_dataset(dataset_name)
    return zip_as_tf_dataset(x_train, y_train), zip_as_tf_dataset(x_test, y_test)


def load_dataset(dataset_name: str) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    path_to_train, path_to_test = paths_to_dataset(dataset_name)
    with open(path_to_train) as file:
        x_train, y_train = read_csv(file)
    with open(path_to_test) as file:
        x_test, y_test = read_csv(file)
    return x_train, y_train, x_test, y_test


def paths_to_dataset(dataset_name: str) -> (str, str):
    path_to_dataset = os.path.join(os.path.pardir, 'data', dataset_name)
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)
    path_to_train = os.path.join(path_to_dataset, 'train.csv')
    path_to_test = os.path.join(path_to_dataset, 'test.csv')
    return path_to_train, path_to_test


def generate_to_file(generator: FKGenerator, path_to_file: str) -> None:
    feat_names = [f"x{str(i)}" for i in range(generator.input_dim)] + \
                 [f"y{str(i)}" for i in range(generator.output_dim)]
    raw_data = []
    for x_batch, y_batch in tqdm(generator):
        for x_sample, y_sample in zip(x_batch, y_batch):
            raw_data.append(np.concatenate([x_sample, y_sample]))
    with open(path_to_file, 'w') as file:
        write_csv(feat_names, raw_data, file)


def write_csv(feat_names: Iterable[str], raw_data: Iterable[Iterable[Any]], file: TextIO):
    writer = csv.writer(file)
    writer.writerow(feat_names)
    writer.writerows(raw_data)


def read_csv(file: TextIO) -> (Iterable[str], Iterable[Iterable[Any]]):
    reader = csv.DictReader(file)

    feat_names = reader.fieldnames
    raw_data = []
    for row in tqdm(reader):
        raw_data.append([float(row[feat]) for feat in feat_names])
    return feat_names, raw_data


def vec_to_frame(vec: np.ndarray) -> Frame:
    return Frame.from_q_4(vec[:4], vec[4:, np.newaxis])


def frame_to_vec(frame: Frame) -> np.ndarray:
    quat = frame.q_4
    tr = frame.t_3_1
    return np.append(quat, tr)
