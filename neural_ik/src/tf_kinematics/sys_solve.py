from typing import Callable, Union

import tensorflow as tf
from keras.optimizers import TFOptimizer


SystemOfEq = Callable[[tf.Variable], tf.Tensor]
LossFn = Callable[[tf.Tensor, tf.Tensor], float]


def iter_grad(y_goal: tf.Tensor, x: Union[tf.Variable, tf.Tensor], sys_fn: SystemOfEq,
              loss_fn: LossFn):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = sys_fn(x)
        loss = loss_fn(y_goal, y)
    grad = tape.gradient(loss, x)
    return grad


def iter_grad_hess(y_goal: tf.Tensor, x: Union[tf.Variable, tf.Tensor], sys_fn: SystemOfEq,
                   loss_fn: LossFn):
    with tf.GradientTape(persistent=True) as hess_tape:
        hess_tape.watch(x)
        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(x)
            y = sys_fn(x)
            loss = loss_fn(y_goal, y)
            grad = grad_tape.gradient(loss, x)
        hess = hess_tape.batch_jacobian(grad, x, experimental_use_pfor=False)
    return grad, hess


def solve_iter(y_goal: tf.Tensor, x: Union[tf.Variable, tf.Tensor], sys_fn: SystemOfEq,
               loss_fn: LossFn, optimizer: TFOptimizer):
    grad = iter_grad(y_goal, x, sys_fn, loss_fn)
    optimizer.apply_gradients([(grad, x)])
    return x


def solve(y_goal: tf.Tensor, x0: tf.Tensor, sys_fn: SystemOfEq,
          n_iters: int, loss_fn: LossFn, optimizer: TFOptimizer) -> tf.Variable:
    x = tf.Variable(initial_value=x0)
    for i in range(n_iters):
        x = solve_iter(y_goal, x, sys_fn, loss_fn, optimizer)
    return x
