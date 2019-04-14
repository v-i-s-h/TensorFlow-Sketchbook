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
from tensorflow.python.training.optimizer import Optimizer, _OptimizableVariable, _TensorProcessor, _DenseResourceVariableProcessor, _DenseResourceVariableProcessor, _get_processor

# class _OptimizableVariable(object):
#   """Interface for abstracting over variables in the optimizers."""

#   def target(self):
#     """Returns the optimization target for this variable."""
#     raise NotImplementedError("Calling an abstract method.")

#   def update_op(self, optimizer, g):
#     """Returns the update ops for updating the variable."""
#     raise NotImplementedError("Calling an abstract method.")


# class _TensorProcessor(_OptimizableVariable):
#     """Processor for ordinary Tensors.
#     Even though a Tensor can't really be updated, sometimes it is useful to
#     compute the gradients with respect to a Tensor using the optimizer. Updating
#     the Tensor is, of course, unsupported.
#     """

#     def __init__(self, v):
#         self._v = v

#     def target(self):
#         return self._v

#     def update_op(self, optimizer, g):
#         raise NotImplementedError("Trying to update a Tensor ", self._v)

# class _DenseResourceVariableProcessor(_OptimizableVariable):
#     """Processor for dense ResourceVariables."""

#     def __init__(self, v):
#         self._v = v

#     def target(self):
#         return self._v

#     def update_op(self, optimizer, g):
#         # pylint: disable=protected-access
#         if isinstance(g, ops.IndexedSlices):
#             if self._v.constraint is not None:
#                 raise RuntimeError(
#                 "Cannot use a constraint function on a sparse variable.")
#             return optimizer._resource_apply_sparse_duplicate_indices(
#                 g.values, self._v, g.indices)
#         update_op = optimizer._resource_apply_dense(g, self._v)
#         if self._v.constraint is not None:
#             with ops.control_dependencies([update_op]):
#                 return self._v.assign(self._v.constraint(self._v))
#         else:
#             return update_op


# def _get_processor(v):
#     """The processor of v."""
#     if context.executing_eagerly():
#         if isinstance(v, ops.Tensor):
#             return _TensorProcessor(v)
#         else:
#             return _DenseResourceVariableProcessor(v)
#     if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  # pylint: disable=protected-access
#         # True if and only if `v` was initialized eagerly.
#         return _DenseResourceVariableProcessor(v)
#     if v.op.type == "VarHandleOp":
#         return _DenseResourceVariableProcessor(v)
#     if isinstance(v, variables.Variable):
#         return _RefVariableProcessor(v)
#     if isinstance(v, ops.Tensor):
#         return _TensorProcessor(v)
#     raise NotImplementedError("Trying to optimize unsupported type ", v)

class SPSA(tf.train.GradientDescentOptimizer):
    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2
    
    def __init__(self, a=0.01, c=0.01, alpha=1.0, gamma=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.num_params = 0
        self.tvars = None
        self.work_graph = tf.get_default_graph()

        # optimizer parameters
        self.a = tf.constant(a, dtype=tf.float32, name = "SPSA_a" )
        self.c = tf.constant(c, dtype=tf.float32, name = "SPSA_c" )
        self.alpha = tf.constant(alpha, dtype=tf.float32, name = "SPSA_alpha" )
        self.gamma = tf.constant(gamma, dtype=tf.float32, name = "SPSA_gamma" )


    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
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

        # print("var_refs:")
        # for vr in var_refs:
        #     print(vr)

        # ==================================================================================
        # grads = gradients.gradients(
        #     loss, var_refs, grad_ys=grad_loss,
        #     gate_gradients=(gate_gradients == SPSA.GATE_OP),
        #     aggregation_method=aggregation_method,
        #     colocate_gradients_with_ops=colocate_gradients_with_ops)
        
        # grads = [ tf.zeros(tf.shape(vrefs)) for vrefs in var_refs ]
        orig_graph_view = None
        trainable_vars = var_list
        # self.tvars = var_list
        self.tvars = [var.name.split(':')[0] for var in var_list]  # list of names of trainable variables
        self.global_step_tensor = tf.Variable(0, name='global_step', trainable=False)

        # Perturbations
        deltas = {}
        n_perturbations = {}
        p_perturbations = {}
        with tf.name_scope("Perturbator"):
            self.c_t = tf.div( self.c,  tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32),
                                              tf.constant(1, dtype=tf.float32)), self.gamma), name = "SPSA_ct" )
            for var in trainable_vars:
                self.num_params += self._mul_dims(var.get_shape())
                var_name = var.name.split(':')[0]
                random = Bernoulli(tf.fill(var.get_shape(), 0.5), dtype=tf.float32)
                deltas[var] = tf.subtract( tf.constant(1, dtype=tf.float32),
                                    tf.scalar_mul(tf.constant(2, dtype=tf.float32),random.sample(1)[0]), name = "SPSA_delta" )
                c_t_delta = tf.scalar_mul( tf.reshape(self.c_t, []), deltas[var] )
                n_perturbations[var_name+'/read:0'] = tf.subtract( var, c_t_delta, name = "perturb_n" )
                p_perturbations[var_name+'/read:0'] = tf.add(var, c_t_delta, name = "perturb_p" )
        # print("{} parameters".format(self.num_params))

        # Evaluator
        with tf.name_scope("Evaluator"):
            orig_graph_view = ge.sgv(tf.get_default_graph())
            _, self.ninfo = self._clone_model(orig_graph_view, n_perturbations, 'N_Eval')
            _, self.pinfo = self._clone_model(orig_graph_view, p_perturbations, 'P_Eval')

        # Weight Updater
        optimizer_ops = []
        grads = []
        with tf.control_dependencies([loss]):
            with tf.name_scope('Updater'):
                a_t = self.a / (tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32),
                                             tf.constant(1, dtype=tf.float32)), self.alpha))
                for var in trainable_vars:
                    l_pos = self.pinfo.transformed( loss )
                    l_neg = self.ninfo.transformed( loss )
                    ghat = (l_pos - l_neg) / (tf.constant(2, dtype=tf.float32) * self.c_t * deltas[var])
                    optimizer_ops.append(tf.assign_sub(var, a_t*ghat))
                    grads.append(ghat)
        print(tf.get_default_graph())

        print("grads")
        for g in grads:
            print(g)


        #===================================================================================
        if gate_gradients == SPSA.GATE_GRAPH:
            print("===================")
            grads = control_flow_ops.tuple(grads)
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars
            if g is not None and v.dtype != dtypes.resource])
        return grads_and_vars

    def _clone_model(self, model, perturbations, dst_scope):
        ''' make a copy of model and connect the resulting sub-graph to
            input ops of the original graph and parameter assignments by
            perturbator.    
        '''
        def not_placeholder_or_trainvar_filter(op):
            # print(op.name)
            if op.type == 'Placeholder':              # evaluation sub-graphs will be fed from original placeholders
                return False
            for var_name in self.tvars:
                if op.name.startswith(var_name):      # remove Some/Var/(read,assign,...) -- will be replaced with perturbations
                    return False
            return True

        ops_without_inputs = ge.filter_ops(model.ops, not_placeholder_or_trainvar_filter)
        # print("ModelOPS=========================")
        # for o in ops_without_inputs:
        #     print(o.name, o.type)
        # remove init op from clone if already present
        try:
            ops_without_inputs.remove(self.work_graph.get_operation_by_name("init"))
        except:
            pass
        clone_sgv = ge.make_view(ops_without_inputs)
        clone_sgv = clone_sgv.remove_unused_ops(control_inputs=True)

        input_replacements = {}
        for t in clone_sgv.inputs:
            if t.name in perturbations.keys():                  # input from trainable var --> replace with perturbation
                input_replacements[t] = perturbations[t.name]
            else:                                               # otherwise take input from original graph
                input_replacements[t] = self.work_graph.get_tensor_by_name(t.name)
        return ge.copy_with_input_replacements(clone_sgv, input_replacements, dst_scope=dst_scope)


    def _mul_dims(self, shape):
        n = 1
        for d in shape:
            n *= d.value
        return n