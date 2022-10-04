from typing import Callable, Union

import tensorflow as tf
from keras.losses import LossFunctionWrapper
from keras.optimizers import TFOptimizer


SystemOfEq = Callable[[tf.Variable], tf.Tensor]


@tf.function
def solve_iter_grad(y_goal: tf.Tensor, x: Union[tf.Variable, tf.Tensor], sys_fn: SystemOfEq,
                    loss_fn: LossFunctionWrapper):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = sys_fn(x)
        loss = loss_fn(y_goal, y)
    grad = tape.gradient(loss, x)
    return grad


def solve_iter(y_goal: tf.Tensor, x: Union[tf.Variable, tf.Tensor], sys_fn: SystemOfEq,
               loss_fn: LossFunctionWrapper, optimizer: TFOptimizer):
    grad = solve_iter_grad(y_goal, x, sys_fn, loss_fn)
    optimizer.apply_gradients([(grad, x)])
    return x


def solve(y_goal: tf.Tensor, x0: tf.Tensor, sys_fn: SystemOfEq,
          n_iters: int, loss_fn: LossFunctionWrapper, optimizer: TFOptimizer) -> tf.Variable:
    x = tf.Variable(initial_value=x0)
    for i in range(n_iters):
        x = solve_iter(y_goal, x, sys_fn, loss_fn, optimizer)
    return x
