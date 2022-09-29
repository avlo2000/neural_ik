from pathlib import Path

from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.urdf import chain_from_urdf_file


PATH_TO_URDF = Path('../urdf').absolute()


def kuka_robot(batch_size: int) -> DLKinematics:
    path_to_urdf = PATH_TO_URDF/'kuka.urdf'
    from_link = 'calib_kuka_arm_base_link'
    to_link = 'kuka_arm_7_link'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link, batch_size=batch_size)
    return kinematics


def human_robot(batch_size: int) -> DLKinematics:
    path_to_urdf = PATH_TO_URDF/'human.urdf'
    from_link = 'human_base'
    to_link = 'human_right_hand'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link, batch_size=batch_size)
    return kinematics


def omnipointer_robot(batch_size: int) -> DLKinematics:
    path_to_urdf = PATH_TO_URDF/'omnipointer.urdf'
    from_link = 'base_link'
    to_link = 'arm_link_5'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link, batch_size=batch_size)
    return kinematics
