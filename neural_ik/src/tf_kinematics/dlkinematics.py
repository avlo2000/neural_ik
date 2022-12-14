from __future__ import absolute_import

import tensorflow as tf

from tf_kinematics.fk_solver import solve_static, solve_forward
from tf_kinematics.urdf_parser.urdf import Robot  # noqa


class DLKinematics:
    def __init__(self, urdf: Robot, base_link, end_link, batch_size=1):
        if not isinstance(urdf, Robot):
            raise DLKinematicsError(
                'Expected urdf to be type {0} got {1}'.format(Robot, type(urdf)))

        self.urdf = urdf
        self.link_names = [link.name for link in urdf.links]
        self.batch_size = batch_size
        self.model_name = urdf.name

        if base_link not in self.link_names:
            raise DLKinematicsError('Link "{0}" not in {1}'.format(base_link, self.link_names))

        if end_link not in self.link_names:
            raise DLKinematicsError('Link "{0}" not in {1}'.format(end_link, self.link_names))

        self.base_link = base_link
        self.end_link = end_link

        chain = urdf.get_chain(base_link, end_link, joints=True, links=False, fixed=True)
        self._chain = [Joint(urdf.joint_map.get(x)) for x in chain]

        self.theta_indices = self.generate_chain_indices()
        self.thetas_shape = tf.cast(
            [len(self._chain), self.batch_size, 6], tf.int32)

        # Keep them for debugging
        self.static_matrices = tf.constant(
            solve_static(self._chain), dtype=tf.float64)

        self.forward_matrices = tf.Variable(
            tf.stack([self.static_matrices] * self.batch_size), dtype=tf.float64)

        # Change dimensions for forward matrices here:
        self.forward_matrices = tf.transpose(
            self.forward_matrices, [1, 0, 2, 3])

    @property
    def dof(self):
        return len(self.get_chain(fixed=False, joints=True, links=False))

    @property
    def limits(self):
        chain = self.get_chain(fixed=False, joints=True, links=False)
        limits = [joint.limit for joint in chain]
        return tf.stack(limits, axis=1)

    def generate_chain_indices(self):
        theta_indices = list()
        for batch in range(self.batch_size):
            for (idx, joint) in enumerate(self._chain):
                if joint.type == 'continuous' or joint.type == 'revolute':
                    theta_indices.append(
                        [idx, batch, joint.axis.index(1)])
                elif joint.type == 'prismatic':
                    theta_indices.append(
                        [idx, batch, joint.axis.index(1) + 3])
                elif joint.type == 'floating':
                    for i in range(6):
                        theta_indices.append(
                            [idx, batch, i])
                elif joint.type == 'planar':
                    # @ToDo: Planar joint has exactly 2 DoF in either x,y or z.
                    for idx, axis in enumerate(joint.axis):
                        if axis == 1.:
                            theta_indices.append(
                                [idx, batch, idx + 3])

                elif joint.type == 'fixed':
                    pass

        # Workaround if there no joints with any DoF
        if len(theta_indices) == 0:
            theta_indices.append([0, 0, 0])
        return tf.cast(theta_indices, tf.int32)

    def get_chain(self, fixed=False, joints=True, links=False):
        t = self.urdf.get_chain(
            self.base_link, self.end_link, joints, fixed, links)
        return [Joint(self.urdf.joint_map.get(x)) for x in t]

    @tf.function
    def forward(self, thetas) -> tf.Tensor:
        return solve_forward(self.forward_matrices, thetas, self.theta_indices, self.thetas_shape)

    @property
    def num_joints(self):
        return len(self.get_chain())

    def __str__(self):
        return '<{0}>'.format(self.__class__)


# Class to transform joint to tensor


class Joint:
    def __init__(self, joint):
        self.joint = joint
        self.type = self.joint_type

    @property
    def axis(self):
        # @ToDo Move this to urdf parse validation
        if self.joint.axis is None:
            raise DLKinematicsError('Joint "{0}" has no property axis')

        self.joint.axis = list(map(abs, self.joint.axis))

        return self.joint.axis

    @property
    def limit(self):
        # @ToDo Set Joint limits to 0.0 if the joint is continious
        return tf.constant([self.joint.limit.lower, self.joint.limit.upper], dtype=tf.float64)

    @property
    def offset(self):
        return tf.constant(self.joint.origin.xyz, dtype=tf.float64)

    @property
    def rotation(self):
        return tf.constant(self.joint.origin.rpy, dtype=tf.float64)

    @property
    def joint_type(self):
        return self.joint.type

    @property
    def name(self):
        return self.joint.name


class DLKinematicsError(Exception):
    def __init__(self, *args):
        if args:
            self.error_message = args[0]
        else:
            self.error_message = None

    def __str__(self):
        if self.error_message:
            return 'DLKinematicsError: {0}'.format(self.error_message)
        else:
            return 'DLKinematicsError'
