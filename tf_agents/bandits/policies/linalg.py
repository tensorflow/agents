# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility code for linear algebra functions."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.typing import types
from tf_agents.utils import common


def _cg_check_shapes(a_mat, b):
  if a_mat.shape[0] != a_mat.shape[1] or a_mat.shape.rank != 2:
    raise ValueError('`a_mat` must be rank 2 square matrix; '
                     'got shape {}.'.format(a_mat.shape))
  if a_mat.shape[1] != b.shape[0]:
    raise ValueError('The dims of `a_mat` and `b` are not compatible; '
                     'got shapes {} and {}.'.format(a_mat.shape, b.shape))


@common.function
def conjugate_gradient(a_mat: types.Tensor,
                       b: types.Tensor,
                       tol: float = 1e-10) -> types.Float:
  """Returns `x` such that `A * x = b`.

  Implements the Conjugate Gradient method.
  https://en.wikipedia.org/wiki/Conjugate_gradient_method

  Args:
    a_mat: a Symmetric Positive Definite matrix, represented as a `Tensor` of
      shape `[n, n]`.
    b: a `Tensor` of shape `[n, 1]`.
    tol: (float) desired tolerance on the residual.

  Returns:
    x: `Tensor` `x` of shape `[n, 1]` such that `A * x = b`.

  Raises:
    ValueError: if `a_mat` is not square or `a_mat` and `b` have incompatible
    shapes.
  """
  _cg_check_shapes(a_mat, b)
  n = tf.shape(b)[0]
  x = tf.zeros_like(b)

  r = b - tf.matmul(a_mat, x)
  p = r
  rs_old = tf.reduce_sum(r * r)
  rs_new = rs_old

  def body_fn(i, x, p, r, rs_old, rs_new):
    """One iteration of CG."""
    a_x_p = tf.matmul(a_mat, p)
    alpha = rs_old / tf.reduce_sum(p * a_x_p)
    x = x + alpha * p
    r = r - alpha * a_x_p
    rs_new = tf.reduce_sum(r * r)
    p = r + (rs_new / rs_old) * p
    rs_old = rs_new
    i = i + 1
    return i, x, p, r, rs_old, rs_new

  def while_exit_cond(i, x, p, r, rs_old, rs_new):
    """Exit the loop when n is reached or when the residual becomes small."""
    del x  # unused
    del p  # unused
    del r  # unused
    del rs_old  # unused
    i_cond = tf.less(i, n)
    residual_cond = tf.greater(tf.sqrt(rs_new), tol)
    return tf.logical_and(i_cond, residual_cond)

  _, x, _, _, _, _ = tf.while_loop(
      while_exit_cond,
      body_fn,
      [tf.constant(0), x, p, r, rs_old, rs_new],
      parallel_iterations=1)
  return x


@common.function
def conjugate_gradient_solve(a_mat: types.Tensor,
                             b_mat: types.Tensor,
                             tol: float = 1e-10) -> types.Tensor:
  """Returns `X` such that `A * X = B`.

  Uses Conjugate Gradient to solve many linear systems of equations with the
  same matrix `a_mat` and multiple right hand sides provided as columns in
  the matrix `b_mat`.

  Args:
    a_mat: a Symmetric Positive Definite matrix, represented as a `Tensor` of
      shape `[n, n]`.
    b_mat: a `Tensor` of shape `[n, k]`.
    tol: (float) desired tolerance on the residual.

  Returns:
    X: `Tensor` `X` of shape `[n, k]` such that `A * X = B`.

  Raises:
    ValueError: if `a_mat` is not square or `a_mat` and `b_mat` have
    incompatible shapes.
  """
  # Allows for flexible shape handling. If the shape is statically known, it
  # will use the first part. If the shape is not statically known, tf.shape()
  # will be used.
  n = tf.compat.dimension_value(b_mat.shape[0]) or tf.shape(b_mat)[0]
  k = tf.compat.dimension_value(b_mat.shape[1]) or tf.shape(b_mat)[1]
  x = tf.zeros_like(b_mat)

  def body_fn(i, x):
    """Solve one linear system of equations with the `i`-th column of b_mat."""
    b_vec = tf.slice(b_mat, begin=[0, i], size=[n, 1])
    x_sol = conjugate_gradient(a_mat, b_vec, tol)
    indices = tf.concat([tf.reshape(tf.range(n, dtype=tf.int32), [n, 1]),
                         i * tf.ones([n, 1], dtype=tf.int32)], axis=-1)
    x = tf.tensor_scatter_nd_update(
        tensor=x, indices=indices, updates=tf.squeeze(x_sol, 1))
    x.set_shape(b_mat.shape)
    i = i + 1
    return i, x

  _, x = tf.while_loop(
      lambda i, _: i < k,
      body_fn,
      loop_vars=[tf.constant(0), x],
      parallel_iterations=10)
  return x


def _check_shapes(a_inv: types.Tensor, u: types.Tensor):
  if a_inv.shape[0] != a_inv.shape[1] or a_inv.shape.rank != 2:
    raise ValueError('`a_inv` must be rank 2 square matrix; '
                     'got shape {}.'.format(a_inv.shape))
  if u.shape.rank != 2:
    raise ValueError('`u` must be rank 2 matrix; '
                     'got shape {}.'.format(u.shape))
  if a_inv.shape[1] != u.shape[1]:
    raise ValueError('`a_inv` and `u` must have shapes [m, m] and [n, m]; '
                     'got shapes {} and {}.'.format(a_inv.shape, u.shape))


def simplified_woodbury_update(a_inv: types.Float,
                               u: types.Float) -> types.Float:
  """Returns `w` such that `inverse(a + u.T.dot(u)) = a_inv + w`.

  Makes use of the Woodbury matrix identity. See
  https://en.wikipedia.org/wiki/Woodbury_matrix_identity.

  **NOTE**: This implementation assumes that a_inv is symmetric. Since it's too
  expensive to check symmetricity, the function silently outputs a wrong answer
  in case `a` is not symmetric.

  Args:
    a_inv: an invertible SYMMETRIC `Tensor` of shape `[m, m]`.
    u: a `Tensor` of shape `[n, m]`.
  Returns:
    A `Tensor` `w` of shape `[m, m]` such that
    `inverse(a + u.T.dot(u)) = a_inv + w`.
  Raises:
    ValueError: if `a_inv` is not square or `a_inv` and `u` have incompatible
    shapes.
  """
  _check_shapes(a_inv, u)
  u_x_a_inv = tf.matmul(u, a_inv)
  capacitance = (
      tf.eye(tf.shape(u)[0], dtype=u.dtype) +
      tf.matmul(u_x_a_inv, u, transpose_b=True))
  return -1. * tf.matmul(
      u_x_a_inv, tf.linalg.solve(capacitance, u_x_a_inv), transpose_a=True)


def update_inverse(a_inv: types.Float, x: types.Float) -> types.Float:
  """Updates the inverse using the Woodbury matrix identity.

  Given a matrix `A` of size d-by-d and a matrix `X` of size k-by-d, this
  function computes the inverse of B = A + X^T X, assuming that the inverse of
  A is available.

  Reference:
  https://en.wikipedia.org/wiki/Woodbury_matrix_identity

  Args:
    a_inv: a `Tensor` of shape [`d`, `d`]. This is the current inverse of `A`.
    x: a `Tensor` of shape [`k`, `d`].

  Returns:
    The update that needs to be added to 'a_inv' to compute the inverse.
    If `x` is empty, a zero matrix is returned.
  """
  batch_size = tf.shape(x)[0]

  def true_fn():
    return tf.zeros_like(a_inv)

  def false_fn():
    return simplified_woodbury_update(a_inv, x)

  a_inv_update = tf.cond(tf.equal(batch_size, 0), true_fn, false_fn)
  return a_inv_update
