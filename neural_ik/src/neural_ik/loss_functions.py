from visual_kinematics.RobotSerial import RobotSerial
from core.evaluate_metrics import distance_as_dual_quat_norm
from keras.losses import Loss
import tensorflow as tf


class DualQuatMormLoss(Loss):

    def __init__(self, robot: RobotSerial):
        super().__init__()
        self.__robot = robot

    def call(self, qs_true, qs_pred):
        dists = []
        for q_true, q_pred in zip(qs_true, qs_pred):
            frame_true = self.__robot.forward(q_true)
            frame_pred = self.__robot.forward(q_pred)
            dists.append(distance_as_dual_quat_norm(frame_true, frame_pred))
        return tf.convert_to_tensor(dists)
