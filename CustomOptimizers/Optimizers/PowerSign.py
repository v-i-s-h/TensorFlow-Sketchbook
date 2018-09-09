# Source: https://towardsdatascience.com/custom-optimizer-in-tensorflow-d5b41f75644a

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

import tensorflow as tf

class PowerSignOptimizer( optimizer.Optimizer ):
    def __init__( self, learning_rate = 0.00002, alpha = 0.05, beta = 50, use_locking = False, name = "PowerSign" ):
        super( PowerSignOptimizer, self ).__init__( use_locking, name )
        self._lr    = learning_rate
        self._alpha = alpha
        self._beta  = beta

        # Tensors
        self._lr_t  = None
        self._alpha_t   = None
        self._beta_t    = None

    def _prepare( self ):
        self._lr_t  = ops.convert_to_tensor( self._lr, name = "learning_rate" )
        self._alpha_t   = ops.convert_to_tensor( self._alpha, name = "alpha" )
        self._beta_t    = ops.convert_to_tensor( self._beta, name = "beta" )

    def _create_slots( self, var_list ):
        # Create slots for first and second moments
        for v in var_list:
            self._zeros_slot( v, "m", self._name )

    def _apply_dense( self, grad, var ):
        lr_t    = math_ops.cast( self._lr_t, var.dtype.base_dtype )
        alpha_t = math_ops.cast( self._alpha_t, var.dtype.base_dtype )
        beta_t  = math_ops.cast( self._beta_t, var.dtype.base_dtype )

        eps = 1e-7

        m   = self.get_slot( var, "m" )
        m_t = m.assign( tf.maximum(beta_t*m+eps,tf.abs(grad)) )

        var_update  = state_ops.assign_sub( var, lr_t*grad*tf.exp(tf.log(alpha_t)*tf.sign(grad)*tf.sign(m_t)) )

        return control_flow_ops.group( *[var_update,m_t] )