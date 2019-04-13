import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.distributions import Bernoulli

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import gradients
from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.framework import dtypes

class _OptimizableVariable(object):
  """Interface for abstracting over variables in the optimizers."""

  def target(self):
    """Returns the optimization target for this variable."""
    raise NotImplementedError("Calling an abstract method.")

  def update_op(self, optimizer, g):
    """Returns the update ops for updating the variable."""
    raise NotImplementedError("Calling an abstract method.")


class _TensorProcessor(_OptimizableVariable):
    """Processor for ordinary Tensors.
    Even though a Tensor can't really be updated, sometimes it is useful to
    compute the gradients with respect to a Tensor using the optimizer. Updating
    the Tensor is, of course, unsupported.
    """

    def __init__(self, v):
        self._v = v

    def target(self):
        return self._v

    def update_op(self, optimizer, g):
        raise NotImplementedError("Trying to update a Tensor ", self._v)

class _DenseResourceVariableProcessor(_OptimizableVariable):
    """Processor for dense ResourceVariables."""

    def __init__(self, v):
        self._v = v

    def target(self):
        return self._v

    def update_op(self, optimizer, g):
        # pylint: disable=protected-access
        if isinstance(g, ops.IndexedSlices):
            if self._v.constraint is not None:
                raise RuntimeError(
                "Cannot use a constraint function on a sparse variable.")
            return optimizer._resource_apply_sparse_duplicate_indices(
                g.values, self._v, g.indices)
        update_op = optimizer._resource_apply_dense(g, self._v)
        if self._v.constraint is not None:
            with ops.control_dependencies([update_op]):
                return self._v.assign(self._v.constraint(self._v))
        else:
            return update_op


def _get_processor(v):
    """The processor of v."""
    if context.executing_eagerly():
        if isinstance(v, ops.Tensor):
            return _TensorProcessor(v)
        else:
            return _DenseResourceVariableProcessor(v)
    if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  # pylint: disable=protected-access
        # True if and only if `v` was initialized eagerly.
        return _DenseResourceVariableProcessor(v)
    if v.op.type == "VarHandleOp":
        return _DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return _RefVariableProcessor(v)
    if isinstance(v, ops.Tensor):
        return _TensorProcessor(v)
    raise NotImplementedError("Trying to optimize unsupported type ", v)

class SPSA(tf.train.GradientDescentOptimizer):
    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        """Compute gradients of `loss` for the variables in `var_list`.
        This is the first part of `minimize()`.  It returns a list
        of (gradient, variable) pairs where "gradient" is the gradient
        for "variable".  Note that "gradient" can be a `Tensor`, an
        `IndexedSlices`, or `None` if there is no gradient for the
        given variable.
        Args:
        loss: A Tensor containing the value to minimize or a callable taking
            no arguments which returns the value to minimize. When eager execution
            is enabled it must be a callable.
        var_list: Optional list or tuple of `tf.Variable` to update to minimize
            `loss`.  Defaults to the list of variables collected in the graph
            under the key `GraphKeys.TRAINABLE_VARIABLES`.
        gate_gradients: How to gate the computation of gradients.  Can be
            `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
        aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: If True, try colocating gradients with
            the corresponding op.
        grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
        Returns:
        A list of (gradient, variable) pairs. Variable is always present, but
        gradient can be `None`.
        Raises:
        TypeError: If `var_list` contains anything else than `Variable` objects.
        ValueError: If some arguments are invalid.
        RuntimeError: If called with eager execution enabled and `loss` is
            not callable.
        @compatibility(eager)
        When eager execution is enabled, `gate_gradients`, `aggregation_method`,
        and `colocate_gradients_with_ops` are ignored.
        @end_compatibility
        """
        if callable(loss):
            with backprop.GradientTape() as tape:
                if var_list is not None:
                    tape.watch(var_list)
                    loss_value = loss()

            if var_list is None:
                var_list = tape.watched_variables()
            # TODO(jhseu): Figure out why GradientTape's gradients don't require loss
            # to be executed.
            with ops.control_dependencies([loss_value]):
                grads = tape.gradient(loss_value, var_list, grad_loss)
            return list(zip(grads, var_list))

        # # Non-callable/Tensor loss case
        # if context.executing_eagerly():
        #     raise RuntimeError(
        #         "`loss` passed to Optimizer.compute_gradients should "
        #         "be a function when eager execution is enabled.")

        if gate_gradients not in [SPSA.GATE_NONE, SPSA.GATE_OP, SPSA.GATE_GRAPH]:
            raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                            "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                            gate_gradients)
        self._assert_valid_dtypes([loss])
        if grad_loss is not None:
            self._assert_valid_dtypes([grad_loss])
        if var_list is None:
            var_list = (
                variables.trainable_variables() +
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        else:
            var_list = nest.flatten(var_list)
        # pylint: disable=protected-access
        var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
        # pylint: enable=protected-access
        processors = [_get_processor(v) for v in var_list]
        if not var_list:
            raise ValueError("No variables to optimize.")
        var_refs = [p.target() for p in processors]
        grads = gradients.gradients(
            loss, var_refs, grad_ys=grad_loss,
            gate_gradients=(gate_gradients == SPSA.GATE_OP),
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops)
        if gate_gradients == SPSA.GATE_GRAPH:
            grads = control_flow_ops.tuple(grads)
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars
            if g is not None and v.dtype != dtypes.resource])
        return grads_and_vars