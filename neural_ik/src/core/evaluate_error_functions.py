import numpy as np
from visual_kinematics import Frame
from dual_quaternions import DualQuaternion


def frame_to_dual_quat(frame: Frame) -> DualQuaternion:
    return DualQuaternion.from_homogeneous_matrix(frame.t_4_4)


def distance_as_dual_quat_norm(frame1: Frame, frame2: Frame) -> float:
    diff_frame = frame1 * frame2.inv
    if np.isclose(diff_frame.t_4_4, np.eye(4)).all():
        return 0.0
    dual_quat = frame_to_dual_quat(diff_frame)

    return dual_quat.q_r.norm + dual_quat.q_d.norm


