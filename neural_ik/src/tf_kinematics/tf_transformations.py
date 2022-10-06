from __future__ import print_function, division, unicode_literals
import tensorflow as tf

"""

Matrix([
    [1.0*cy*cz, -1.0*cx*sz + cz*sx*sy, cx*cz*sy + 1.0*sx*sz,   0],
    [1.0*cy*sz,  1.0*cx*cz + sx*sy*sz, cx*sy*sz - 1.0*cz*sx,   0],
    [-1.0*sy,             1.0*cy*sx,            1.0*cx*cy,   0],
    [0,                     0,                    0, 1.0]])
"""


@tf.function
def tf_homogeneous_transformation(sin, cos, translation):
    sx, sy, sz = tf.unstack(sin, num=3, axis=-1)
    cx, cy, cz = tf.unstack(cos, num=3, axis=-1)
    x, y, z = tf.unstack(translation, num=3, axis=-1)
    zeros = tf.zeros(tf.shape(sx), dtype=tf.float32)
    ones = tf.ones(tf.shape(sx), dtype=tf.float32)
    r00 = cy * cz
    r01 = -1.0 * cx * sz + cz * sx * sy
    r02 = cx * cz * sy + 1.0 * sx * sz
    r10 = cy * sz
    r11 = cx * cz + sx * sy * sz
    r12 = cx * sy * sz - 1.0 * cz * sx
    r20 = -1.0 * sy
    r21 = cy * sx
    r22 = cx * cy

    transformation_matrix = tf.stack((
        r00, r01, r02, x,
        r10, r11, r12, y,
        r20, r21, r22, z,
        zeros, zeros, zeros, ones
    ), axis=-1)
    output_shape = tf.concat((tf.shape(sin)[:-1], (4, 4)), axis=-1)
    return tf.reshape(transformation_matrix, shape=output_shape)


# @tf.function(input_signature=[tf.TensorSpec(shape=(None, 4, 4))])
def tf_compact(transformation: tf.Tensor):
    trans = transformation[:, :3, 3]
    rot = transformation[:, :3, :3]
    angle_axis = tf_rot_to_angle_axis(rot)
    return tf.concat((trans, angle_axis), axis=1)


# @tf.function
def tf_rot_to_angle_axis(rot: tf.Tensor):
    angle = tf.acos(tf.clip_by_value((tf.linalg.trace(rot[:]) - 1.0) / 2.0, clip_value_min=-1.0, clip_value_max=1.0))
    angle_axis = (rot[:, 2, 1] - rot[:, 1, 2],
                  rot[:, 0, 2] - rot[:, 2, 0],
                  rot[:, 1, 0] - rot[:, 0, 1])
    angle_axis = tf.convert_to_tensor(angle_axis)
    angle = tf.expand_dims(angle, axis=-1)
    return angle * tf.transpose(angle_axis, [1, 0]) / (2 * tf.sin(angle) + 0.000001)


@tf.function
def tf_dist_l2(t1: tf.Tensor, t2: tf.Tensor):
    diff = tf.matmul(tf.linalg.inv(t1), t2)
    return tf.norm(diff)
