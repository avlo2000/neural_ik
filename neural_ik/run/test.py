
import tensorflow as tf

from neural_ik.models import fk_dnn
from tf_kinematics.kinematic_models import kuka_robot


if __name__ == '__main__':
    kin = kuka_robot()
    model = fk_dnn(kin)
    print(model(tf.random.uniform(shape=(1, kin.dof))))
