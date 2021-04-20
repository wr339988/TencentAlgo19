import tensorflow as tf
from src import misc_utils as utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from models.base_model import BaseModel
import numpy as np
import time 
import os
class Model(BaseModel):
    def __init__(self,hparams):
        self.hparams=hparams
        if hparams.metric in ['SMAPE']:
            self.best_score=100000
        else:
            self.best_score=0
        self.build_graph(hparams)   
        self.optimizer(hparams)

              
        params = tf.trainable_variables()
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))   
  
    def set_Session(self,sess):
        self.sess=sess
        
    def build_graph(self, hparams):
        self.initializer = self._get_initializer(hparams)
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        self.use_norm=tf.placeholder(tf.bool)
        hparams.feature_nums=len(hparams.cross_features)
        emb_inp_v2=[]
        dnn_input=[]
        
        
        if hparams.dense_features is not None:
            self.dense_features=tf.placeholder(shape=(None,len(hparams.dense_features)), dtype=tf.float32)
        
        #key-values memory, add to CIN    
        if hparams.kv_features is not None:
            self.kv_features=tf.placeholder(shape=(None,len(hparams.kv_features)), dtype=tf.float32)
            kv_emb_v2=tf.get_variable(shape=[len(hparams.kv_features),hparams.kv_batch_num+1,hparams.k],
                                                  initializer=self.initializer,name='emb_v2_kv')
            index=[i/hparams.kv_batch_num for i in range(hparams.kv_batch_num+1)]    
            index=tf.constant(index)  
            distance=1/(tf.abs(self.kv_features[:,:,None]-index[None,None,:])+0.00001)
            weights=tf.nn.softmax(distance,-1) #[batch_size,kv_features_size,kv_batch_num]
            kv_emb=tf.reduce_sum(weights[:,:,:,None]*kv_emb_v2[None,:,:,:],-2)
            
            hparams.feature_nums+=len(hparams.kv_features)
            emb_inp_v2.append(kv_emb)
            
            kv_emb=tf.reshape(kv_emb,[-1,len(hparams.kv_features)*hparams.k])
            dnn_input.append(kv_emb)
               
        if hparams.multi_features is not None:
            hparams.feature_nums+=len(hparams.multi_features)
            
        #CIN: 结合多值特征和交叉特征输入CIN中 
        if hparams.cross_features is not None:    
            self.cross_features=tf.placeholder(shape=(None,len(hparams.cross_features)), dtype=tf.int32)
            self.cross_emb_v2=tf.get_variable(shape=[hparams.cross_hash_num,hparams.k],initializer=self.initializer,name='emb_v2_cross')
            emb_inp_v2.append(tf.gather(self.cross_emb_v2, self.cross_features))
        
        if hparams.multi_features is not None:
            self.multi_features=tf.placeholder(shape=(None,len(hparams.multi_features),None), dtype=tf.int32)
            self.multi_weights=tf.placeholder(shape=(None,len(hparams.multi_features),None), dtype=tf.float32)
            self.multi_emb_v2=tf.get_variable(shape=[hparams.multi_hash_num,hparams.k],
                                                     initializer=self.initializer,name='emb_v2_multi') 
            emb_multi_v2=tf.gather(self.multi_emb_v2, self.multi_features)
            self.weights=self.multi_weights/(tf.reduce_sum(self.multi_weights,-1)+1e-20)[:,:,None]
            emb_multi_v2=tf.reduce_sum(emb_multi_v2*self.weights[:,:,:,None],2)
            emb_inp_v2.append(emb_multi_v2)
            
        if len(emb_inp_v2)!=0:
            emb_inp_v2=tf.concat(emb_inp_v2,1)
            result=self._build_extreme_FM(hparams, emb_inp_v2, res=False, direct=False, bias=False, reduce_D=False, f_dim=2)
            dnn_input.append(tf.reshape(emb_inp_v2,[-1,hparams.feature_nums*hparams.k]))
            dnn_input.append(result)
        
        #单值特征，直接embedding
        if hparams.single_features is not None:
            self.single_features=tf.placeholder(shape=(None,len(hparams.single_features)), dtype=tf.int32)
            self.single_emb_v2=tf.get_variable(shape=[hparams.single_hash_num,hparams.single_k],
                                               initializer=self.initializer,name='emb_v2_single') 
            dnn_input.append(tf.reshape(tf.gather(self.single_emb_v2, self.single_features),[-1,len(hparams.single_features)*hparams.single_k]))
        
        #稠密特征，这里主要放embedding特征，即word2vec和deepwalk
        if hparams.dense_features is not None:
            dnn_input.append(self.dense_features)
            
        #MLP
        dnn_input=tf.concat(dnn_input,1)
        if hparams.norm is True:
            dnn_input=self.batch_norm_layer(dnn_input,self.use_norm,'dense_norm')        

        input_size=int(dnn_input.shape[-1])
        for idx in range(len(hparams.hidden_size)):
            dnn_input=tf.cond(self.use_norm, lambda: tf.nn.dropout(dnn_input,1-hparams.dropout), lambda: dnn_input)
            glorot = np.sqrt(2.0 / (input_size + hparams.hidden_size[idx]))
            W = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, hparams.hidden_size[idx])), dtype=np.float32)
            dnn_input=tf.tensordot(dnn_input,W,[[-1],[0]])
            dnn_input=tf.nn.relu(dnn_input)
            input_size=hparams.hidden_size[idx]

        glorot = np.sqrt(2.0 / (hparams.hidden_size[-1] + 1))
        W = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(hparams.hidden_size[-1], 1)), dtype=np.float32)     
        logit=tf.tensordot(dnn_input,W,[[-1],[0]])

        #logit = tf.add(logit, self._build_dnn(hparams, embed_out, embed_layer_size))
        logit = tf.add(logit, self._build_linear(hparams))

        self.val=logit[:,0]

        self.score=tf.abs(self.val-self.label)
        self.loss=tf.reduce_mean(self.score)
        self.saver= tf.train.Saver()


    def _build_linear(self, hparams):        
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[len(hparams.single_features)*hparams.single_k, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.reshape(tf.gather(self.single_emb_v2, self.single_features),[-1,len(hparams.single_features)*hparams.single_k])
            linear_output = tf.add(tf.matmul(x, w_linear), b_linear)
            #self.layer_params.append(w_linear)
            #self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output


    def _build_AutoInt(self, hparams, nn_input, num_units=None, num_heads=1, dropout_keep_prob=1, is_training=True, has_residual=True):        
        if num_units is None:
            num_units = nn_input.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(nn_input, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(nn_input, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(nn_input, num_units, activation=tf.nn.relu)
        if has_residual:
            V_res = tf.layers.dense(nn_input, num_units, activation=tf.nn.relu)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

        # Activation
        weights = tf.nn.softmax(weights)

        # Dropouts
        weights = tf.layers.dropout(weights, rate=1-dropout_keep_prob, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(weights, V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        if has_residual:
            outputs += V_res
        
        outputs = tf.nn.relu(outputs)
        # Normalize
        outputs = utils.normalize(outputs)
        
        return outputs
         
    def _build_extreme_FM(self, hparams, nn_input, res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.feature_nums
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.k])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.k * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.k * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.k, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                if reduce_D:
                    filters0 = tf.get_variable("f0_" + str(idx),
                                               shape=[1, layer_size, field_nums[0], f_dim],
                                               dtype=tf.float32)
                    filters_ = tf.get_variable("f__" + str(idx),
                                               shape=[1, layer_size, f_dim, field_nums[-1]],
                                               dtype=tf.float32)
                    filters_m = tf.matmul(filters0, filters_)
                    filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                    filters = tf.transpose(filters_o, perm=[0, 2, 1])
                else:
                    filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)
                # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                
                # BIAS ADD
                if bias:
                    b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                    curr_out = tf.nn.bias_add(curr_out, b)

                curr_out = self._activate(curr_out, hparams.cross_activation)
                
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
                
                if direct:

                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)


            result = tf.concat(final_result, axis=1)
            
            result = tf.reduce_sum(result, -1)

            return result

    
    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        """
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        last_layer_size = hparams.FIELD_COUNT * hparams.dim
        """
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                     curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                     curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                                 w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                                 b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
