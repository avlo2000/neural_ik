import numpy as np
from tqdm import tqdm
from typing import Tuple, Iterable, Any

from visual_kinematics import Frame

from data.abstract_generator import FKGenerator
import csv


def generate_to_file(generator: FKGenerator, path_to_file: str) -> None:
    feat_names = [f"x{str(i)}" for i in range(generator.input_dim)] + \
                 [f"y{str(i)}" for i in range(generator.output_dim)]
    raw_data = []
    for x_batch, y_batch in tqdm(generator):
        for x_sample, y_sample in zip(x_batch, y_batch):
            raw_data.append(np.concatenate([x_sample, y_sample]))
    write(feat_names, raw_data, path_to_file)


def write(feat_names: Iterable[str], raw_data: Iterable[Iterable[Any]], path_to_file: str):
    with open(path_to_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(feat_names)
        for row in raw_data:
            writer.writerow(row)


def read(path_to_file: str) -> Tuple[list, list]:
    with open(path_to_file, 'r') as file:
        reader = csv.DictReader(file)

        feat_names = next(reader)
        x = []
        y = []
        for row in reader:
            x.append(np.asarray([float(row[feat]) for feat in feat_names if feat.startswith('x')]))
            y.append(np.asarray([float(row[feat]) for feat in feat_names if feat.startswith('y')]))
    return x, y


def vec_to_frame(vec: np.ndarray) -> Frame:
    return Frame.from_q_4(vec[:4], vec[4:, np.newaxis])


def frame_to_vec(frame: Frame) -> np.ndarray:
    quat = frame.q_4
    tr = frame.t_3_1
    return np.append(quat, tr)
