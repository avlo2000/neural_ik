from pathlib import Path
from typing import Callable

from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.urdf import chain_from_urdf_file

PATH_TO_URDF = Path('../urdf').absolute()
_KINEMATIC_MODELS_REGISTER = dict()


def kinematic_model(func: Callable):
    _KINEMATIC_MODELS_REGISTER[func.__name__] = func


def load(model_name: str, batch_size: int) -> DLKinematics:
    assert model_name in _KINEMATIC_MODELS_REGISTER, f"{model_name} does not have loading method"
    return _KINEMATIC_MODELS_REGISTER[model_name](batch_size)


@kinematic_model
def kuka_robot(batch_size: int) -> DLKinematics:
    path_to_urdf = PATH_TO_URDF/'kuka.urdf'
    from_link = 'calib_kuka_arm_base_link'
    to_link = 'kuka_arm_7_link'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link, batch_size=batch_size)
    return kinematics


@kinematic_model
def human_robot(batch_size: int) -> DLKinematics:
    path_to_urdf = PATH_TO_URDF/'human.urdf'
    from_link = 'human_base'
    to_link = 'human_right_hand'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link, batch_size=batch_size)
    return kinematics


@kinematic_model
def omnipointer_robot(batch_size: int) -> DLKinematics:
    path_to_urdf = PATH_TO_URDF/'omnipointer.urdf'
    from_link = 'base_link'
    to_link = 'arm_link_5'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link, batch_size=batch_size)
    return kinematics

