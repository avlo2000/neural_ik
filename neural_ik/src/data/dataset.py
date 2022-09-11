import numpy as np
from tqdm import tqdm
from typing import Tuple

from visual_kinematics import Frame

from data.abstract_generator import FKGenerator
import csv


def generate_to_file(generator: FKGenerator, path_to_file: str) -> None:
    file = open(path_to_file, 'w')
    writer = csv.writer(file)

    feat_names = [f"x{str(i)}" for i in range(generator.input_dim)] + \
                 [f"y{str(i)}" for i in range(generator.output_dim)]
    writer.writerow(feat_names)
    for x_batch, y_batch in tqdm(generator):
        for x_sample, y_sample in zip(x_batch, y_batch):
            writer.writerow(np.concatenate([x_sample, y_sample]))


def read(path_to_file: str) -> Tuple[list, list]:
    file = open(path_to_file, 'r')
    reader = csv.DictReader(file)

    feat_names = next(reader)
    x = []
    y = []
    for row in reader:
        x.append(np.asarray([float(row[feat]) for feat in feat_names if feat.startswith('x')]))
        y.append(np.asarray([float(row[feat]) for feat in feat_names if feat.startswith('y')]))
    return x, y


def vec_to_frame(vec: np.ndarray) -> Frame:
    return Frame.from_euler_3(vec[3:, np.newaxis], vec[:3, np.newaxis])
