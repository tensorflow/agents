# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility code for linear algebra functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.typing import types
from tf_agents.utils import common


def _cg_check_shapes(a_mat, b_mat):
  if a_mat.shape[0] != a_mat.shape[1] or a_mat.shape.rank != 2:
    raise ValueError('`a_mat` must be rank 2 square matrix; '
                     'got shape {}.'.format(a_mat.shape))
  if a_mat.shape[1] != b_mat.shape[0]:
    raise ValueError('The dims of `a_mat` and `b_mat` are not compatible; '
                     'got shapes {} and {}.'.format(a_mat.shape, b_mat.shape))


@common.function
def conjugate_gradient(a_mat: types.Tensor,
                       b_mat: types.Tensor,
                       tol: float = 1e-10) -> types.Float:
  """Returns `X` such that `A * X = B`.

  Implements the Conjugate Gradient method.
  https://en.wikipedia.org/wiki/Conjugate_gradient_method

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
  _cg_check_shapes(a_mat, b_mat)
  n = tf.shape(b_mat)[0]
  x = tf.zeros_like(b_mat)

  r = b_mat - tf.matmul(a_mat, x)
  p = r
  rs_old = tf.reduce_sum(r * r, axis=0)
  rs_new = rs_old

  def body_fn(i, x, p, r, rs_old, rs_new):
    """One iteration of CG."""
    # Create a boolean mask shaped as [k] indicating the active columns, i.e.,
    # columns corresponding to large residuals. We only update variables
    # corresponding to those columns to avoid numerical issues.
    active_columns_mask = (rs_old > tol)
    # Replicate the mask along axis 0 to be of shape [n, k].
    active_columns_tiled_mask = tf.tile(
        tf.expand_dims(active_columns_mask, axis=0),
        multiples=[tf.shape(b_mat)[0], 1])
    a_x_p = tf.matmul(a_mat, p)
    alpha_diag = tf.linalg.diag(rs_old / tf.reduce_sum(p * a_x_p, axis=0))
    x = tf.where(active_columns_tiled_mask, x + tf.matmul(p, alpha_diag), x)
    r = tf.where(active_columns_tiled_mask, r - tf.matmul(a_x_p, alpha_diag), r)
    rs_new = tf.where(active_columns_mask, tf.reduce_sum(r * r, axis=0), rs_new)
    p = tf.where(active_columns_tiled_mask,
                 r + tf.matmul(p, tf.linalg.diag(rs_new / rs_old)), p)
    rs_old = tf.where(active_columns_mask, rs_new, rs_old)
    i = i + 1
    return i, x, p, r, rs_old, rs_new

  def while_exit_cond(i, x, p, r, rs_old, rs_new):
    """Exit the loop when n is reached or when the residual becomes small."""
    del x  # unused
    del p  # unused
    del r  # unused
    del rs_old  # unused
    i_cond = tf.less(i, n)
    residual_cond = tf.greater(tf.reduce_max(tf.sqrt(rs_new)), tol)
    return tf.logical_and(i_cond, residual_cond)

  _, x, _, _, _, _ = tf.while_loop(
      while_exit_cond,
      body_fn,
      [tf.constant(0), x, p, r, rs_old, rs_new],
      parallel_iterations=1)
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
