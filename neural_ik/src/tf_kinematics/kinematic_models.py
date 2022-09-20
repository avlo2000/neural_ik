from tf_kinematics.dlkinematics import DLKinematics
from tf_kinematics.urdf import chain_from_urdf_file


def kuka_robot():
    path_to_urdf = r'C:\Users\Pavlo\source\repos\math\neural_ik\urdf\kuka.urdf'
    from_link = 'calib_kuka_arm_base_link'
    to_link = 'kuka_arm_7_link'
    chain = chain_from_urdf_file(path_to_urdf)
    kinematics = DLKinematics(chain, from_link, to_link)
    return kinematics
