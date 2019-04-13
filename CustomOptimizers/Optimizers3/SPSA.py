import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.distributions import Bernoulli

class SimultaneousPerturbationOptimizer(tf.train.Optimizer):

    def __init__(self, inputs=None, graph=None, a=0.01, c=0.01, alpha=1.0, gamma=0.4, use_locking=False, name="SPSA"):
        super(SimultaneousPerturbationOptimizer, self).__init__(use_locking, name)
        self.work_graph = tf.get_default_graph() if graph is None else graph                                # the graph to work on
        self.tvars = [var.name.split(':')[0] for var in tf.trainable_variables()]  # list of names of trainable variables
        self.inputs = inputs
        self.num_params = 0

        # optimizer parameters
        self.a = tf.constant(a, dtype=tf.float32, name = "SPSA_a" )
        self.c = tf.constant(c, dtype=tf.float32, name = "SPSA_c" )
        self.alpha = tf.constant(alpha, dtype=tf.float32, name = "SPSA_alpha" )
        self.gamma = tf.constant(gamma, dtype=tf.float32, name = "SPSA_gamma" )

        print( "SPSA:   a = {:4.3f}  c = {:4.3f}  alpha = {:4.3f}  gamma = {:4.3f}".format(a,c,alpha,gamma) )
        print( "Trainable Variables:" )
        for var in tf.trainable_variables():
            var_name = var.name.split(':')[0]
            print( "    {} => {}".format(var.name,var_name) )


    def _clone_model(self, model, perturbations, dst_scope):
        ''' make a copy of model and connect the resulting sub-graph to
            input ops of the original graph and parameter assignments by
            perturbator.    
        '''
        def not_placeholder_or_trainvar_filter(op):
            if op.type == 'Placeholder':              # evaluation sub-graphs will be fed from original placeholders
                return False
            for var_name in self.tvars:
                if op.name.startswith(var_name):      # remove Some/Var/(read,assign,...) -- will be replaced with perturbations
                    return False
            return True

        ops_without_inputs = ge.filter_ops(model.ops, not_placeholder_or_trainvar_filter)
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


    def minimize( self, loss, var_list = None, global_step = None ):
        orig_graph_view = None
        trainable_vars = var_list if var_list != None else tf.trainable_variables()
        if self.inputs is not None:
            seed_ops = [t.op for t in self.inputs]
            result = list(seed_ops)
            wave = set(seed_ops)
            while wave:                 # stolen from grap_editor.select
                new_wave = set()
                for op in wave:
                    for new_t in op.outputs:
                        if new_t == loss:
                            continue
                        for new_op in new_t.consumers():
                            #if new_op not in result and is_within(new_op):
                            if new_op not in result:
                                new_wave.add(new_op)
                for op in new_wave:
                    if op not in result:
                        result.append(op)
                wave = new_wave
            orig_graph_view = ge.sgv(result)
        else:
            orig_graph_view = ge.sgv(self.work_graph)

        self.global_step_tensor = tf.Variable(0, name='global_step', trainable=False) if global_step is None else global_step

        # Perturbations
        deltas = {}
        n_perturbations = {}
        p_perturbations = {}
        with tf.name_scope("Perturbator"):
            self.c_t = tf.div( self.c,  tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32),
                                              tf.constant(1, dtype=tf.float32)), self.gamma), name = "SPSA_ct" )
            # self.c_t = 0.00 #MOD
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
            _, self.ninfo = self._clone_model(orig_graph_view, n_perturbations, 'N_Eval')
            _, self.pinfo = self._clone_model(orig_graph_view, p_perturbations, 'P_Eval')

        # Weight Updater
        optimizer_ops = []
        with tf.control_dependencies([loss]):
            with tf.name_scope('Updater'):
                a_t = self.a / (tf.pow(tf.add(tf.cast(self.global_step_tensor, tf.float32),
                                             tf.constant(1, dtype=tf.float32)), self.alpha))
                # a_t = 0.00 #MOD
                for var in trainable_vars:
                    l_pos = self.pinfo.transformed( loss )
                    l_neg = self.ninfo.transformed( loss )
                    # print( "l_pos: ", l_pos)
                    # print( "l_neg: ", l_neg)
                    ghat = (l_pos - l_neg) / (tf.constant(2, dtype=tf.float32) * self.c_t * deltas[var])
                    optimizer_ops.append(tf.assign_sub(var, a_t*ghat))
        grp = control_flow_ops.group(*optimizer_ops)
        with tf.control_dependencies([grp]):
             tf.assign_add(self.global_step_tensor, tf.constant(1, dtype=self.global_step_tensor.dtype))

        return grp


    def get_clones(self, op):
        return (self.ninfo.transformed(op), self.pinfo.transformed(op))



# ======================================================================================================================
class MyOpt(tf.train.GradientDescentOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
