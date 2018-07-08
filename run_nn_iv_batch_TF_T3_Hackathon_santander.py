import os, sys, cPickle, time, shutil, logging
sys.stdout.flush()
import math, numpy, scipy, random
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.lstm_ops import LSTMBlockFusedCell
from modules import make_logger
import cPickle
# import numpy as np
logger = make_logger("Main")

min_max_file_name="/home/dawna/tts/mw545/DVExp/hh/min_max.dat"
logger.info('loading min max file '+min_max_file_name)
min_max = cPickle.load(open(min_max_file_name, 'rb'))
y_diff = (min_max["y_max"] - min_max["y_min"])

data_norm_file_name="/home/dawna/tts/mw545/DVExp/hh/data_norm.dat"
logger.info('loading normalised data file '+data_norm_file_name)
data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
logger.info(data_norm["x_train"].shape)
logger.info(data_norm["y_train"].shape)

class acoustic_model_cfg(object):
    def __init__(self, work_dir="/home/dawna/tts/mw545/DVExp/hh"):

        self.input_dim  = 21
        # self.input_dim  = 4735 * 2

        self.batch_size = 256
        self.learning_rate = 0.001
        self.num_epoch     = 2000
        # self.rnn_layer_type = ['LSTM'] * 4
        self.rnn_layer_type = ['Tanh'] * 10
        # self.rnn_layer_type = ['TANH', 'LSTM', 'LSTM', 'TANH', 'TANH']
        # self.high_way = [False, False, False, True, True]
        self.high_way = [False, True, True, True, True, True, True, True, True, True]
        # self.rnn_layer_size = [ 128 ] * 5
        self.rnn_layer_size = [ 1024 ] * 10
        self.num_rnn_layers = len(self.rnn_layer_type)
        assert self.num_rnn_layers == len(self.rnn_layer_size)
        self.gpu_id = 0
        self.gpu_per_process_gpu_memory_fraction = 0.8

        self.output_dim = 1

        exp_dir = "exp_dnn_"+str(self.num_rnn_layers)
        for i in range(self.num_rnn_layers):
            exp_dir = exp_dir + '_' + self.rnn_layer_type[i] + str(self.rnn_layer_size[i]) + str(self.high_way[i])[0]
        self.exp_dir = os.path.join(work_dir, exp_dir)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        # Copy python file
        shutil.copyfile(os.path.join(work_dir, 'run_nn_iv_batch_TF_T3_DV.py'), os.path.join(self.exp_dir, 'run_nn_iv_batch_TF_T3_DV.py'))
        nnets_file_name = "Model"
        self.nnets_file_name = os.path.join(self.exp_dir, nnets_file_name)

        # if False:
        if True:
            self.TRAINDNN  = True
            self.DNNGEN    = True
        else:
            self.TRAINDNN  = False
            self.DNNGEN    = False

class build_acoustic_model(object):

    def __init__(self, amcfg):

        with tf.device('/device:GPU:'+str(amcfg.gpu_id)):
            self.am_input    = tf.placeholder(tf.float32, shape=[amcfg.batch_size, amcfg.input_dim])
            self.am_target   = tf.placeholder(tf.float32, shape=[amcfg.batch_size, amcfg.output_dim])

            self.output_list = []
            self.rnn_layers = {}
            self.rnn_layer_output  = {}
            self.rnn_layer_c_h = {}
            self.rnn_init_c    = {}
            self.rnn_init_h    = {}
            self.learning_rate = amcfg.learning_rate
            self.learning_rate_holder = tf.placeholder(dtype=tf.float32, name='acoustic_learning_rate_holder')

            for i in range(amcfg.num_rnn_layers):
                with tf.variable_scope('acoustic_layer_'+str(i)):
                    if i == 0:
                        # input layer
                        layer_input = self.am_input
                    else:
                        layer_input = self.rnn_layer_output[i-1]

                    if amcfg.rnn_layer_type[i] == 'Tanh':
                        self.rnn_layer_output[i] = tf.contrib.layers.fully_connected(inputs=layer_input, num_outputs=amcfg.rnn_layer_size[i], activation_fn=tf.nn.tanh)
                    elif amcfg.rnn_layer_type[i] == 'Relu':
                        self.rnn_layer_output[i] = tf.contrib.layers.fully_connected(inputs=layer_input, num_outputs=amcfg.rnn_layer_size[i], activation_fn=tf.nn.relu)

                    if amcfg.high_way[i]:
                        assert amcfg.rnn_layer_size[i] == amcfg.rnn_layer_size[i-1]
                        self.rnn_layer_output[i] = self.rnn_layer_output[i] + layer_input
                        
                # rnn_layer_output.append(layer_output_temp)
                # rnn_layer_state.append(layer_state_temp)
            with tf.variable_scope('acoustic_final_layer'):
                layer_input = self.rnn_layer_output[amcfg.num_rnn_layers-1]
                self.final_layer_output = tf.contrib.layers.fully_connected(inputs=layer_input, num_outputs=amcfg.output_dim, activation_fn=None)

                self.zero_output = tf.zeros(shape=[amcfg.batch_size, amcfg.output_dim], dtype=tf.float32)
                self.denorm_labels  = tf.maximum((self.am_target 
                    + tf.constant(float(1.), dtype=tf.float32)) * tf.constant(float(y_diff/2.), dtype=tf.float32) + tf.constant(min_max["y_min"], dtype=tf.float32), self.zero_output)
                self.denorm_predict = tf.maximum((self.final_layer_output 
                    + tf.constant(float(1.), dtype=tf.float32)) * tf.constant(float(y_diff/2.), dtype=tf.float32) + tf.constant(min_max["y_min"], dtype=tf.float32), self.zero_output)

                self.log_labels  = tf.log( self.denorm_labels  + tf.constant(float(1.), dtype=tf.float32))
                self.log_predict = tf.log( self.denorm_predict + tf.constant(float(1.), dtype=tf.float32))

                self.loss = tf.losses.mean_squared_error(labels=self.log_labels, predictions=self.log_predict)

                # self.loss  = tf.losses.mean_squared_error(labels=self.am_target, predictions=self.final_layer_output)
                # self.accuracy = tf.count_nonzero(tf.greater(self.am_target * self.final_layer_output, 0.), dtype=tf.float32) * tf.constant(float(1./amcfg.batch_size), name="1_/batch_size", dtype=tf.float32)

            self.train_step  = tf.train.AdamOptimizer(learning_rate=self.learning_rate_holder,epsilon=1.e-03).minimize(self.loss)

            # init = tf.initialize_all_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()           
                
            self.output_list.append(self.loss)
            # self.output_list.append(self.accuracy)

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

def config_tf(amcfg):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = amcfg.gpu_per_process_gpu_memory_fraction
    tf_config.allow_soft_placement = True
    tf_config.log_device_placement = True
    return tf_config

def make_a_batch(amcfg, data_x, data_y, batch_size):
    x = numpy.zeros((amcfg.batch_size, amcfg.input_dim))
    y = numpy.zeros((amcfg.batch_size, amcfg.output_dim))
    T = data_x.shape[0]
    batch_list = random.sample(range(T), batch_size)
    for j in range(len(batch_list)):
        i = batch_list[j]
        x[j, :] = data_x[i, :]
        y[j, 0] = data_y[i]
        # if data_y[i+num_frames-1] > 0:
        #     y[j, 0] = 1.
        # else:
        #     y[j, 1] = 1.
    return x, y

def train_NN_iv():

    amcfg = acoustic_model_cfg()
    # amcfg.iv_dim = iv_size
    logger.info('config tensorflow')
    tf_config = config_tf(amcfg)
    logger.info('building model')
    rnn_model = build_acoustic_model(amcfg)
    logger.info('start running model')
    sess = tf.Session(config=tf_config)
    # tf.global_variables_initializer().run()    
    sess.run(rnn_model.init)
    # rnn_model.saver.restore(sess, nnets_file_name)

    early_stop = 0
    epoch = 0
    num_roll_back = 0
    best_validation_loss = sys.float_info.max
    previous_valid_loss  = sys.float_info.max
    num_batch = {
        "train": int(4459/amcfg.batch_size),
        # "test":  int(44812/amcfg.batch_size)
    }

    training_epochs  = 1000
    early_stop_epoch = 5
    nnets_file_name = amcfg.nnets_file_name
    
    while (epoch < training_epochs):
        epoch = epoch + 1
        epoch_start_time = time.time()

        logger.info('start training Epoch '+str(epoch))
        
        for batch_idx in range(num_batch["train"]):
            x, y = make_a_batch(amcfg, data_norm["x_train"], data_norm["y_train"], amcfg.batch_size)
            feed_dict = {}
            feed_dict[rnn_model.am_input]  = x
            feed_dict[rnn_model.am_target] = y
            feed_dict[rnn_model.learning_rate_holder] = rnn_model.learning_rate
            sess.run(rnn_model.train_step, feed_dict=feed_dict)
            # if batch_idx % (num_train_batch/5) == 0 and batch_idx > 0:
                # logger.info('finished training '+str(batch_idx)+', loss is '+str(previous_train_output[-1])+', num frames is '+str(num_frames))

        epoch_train_time = time.time()

        logger.info('start evaluating Epoch '+str(epoch))

        output_string = 'epoch '+str(epoch)
        for error_name in ['train']:
            epoch_loss = []
            # epoch_accuracy = []
            for batch_idx in range(num_batch[error_name]):
                x, y = make_a_batch(amcfg, data_norm["x_"+error_name], data_norm["y_"+error_name], amcfg.batch_size)
                feed_dict = {}
                feed_dict[rnn_model.am_input]  = x
                feed_dict[rnn_model.am_target] = y
                feed_dict[rnn_model.learning_rate_holder] = rnn_model.learning_rate
                batch_loss = sess.run(fetches=rnn_model.output_list, feed_dict=feed_dict)
                epoch_loss.append(batch_loss)
            epoch_rmsle_loss = numpy.sqrt(numpy.mean(epoch_loss))
            output_string = output_string + ', '+error_name+' loss is '+str(epoch_rmsle_loss)#+', accuracy is '+str(numpy.mean(epoch_accuracy))

            # logger.info('epoch '+str(epoch)+', loss of '+error_name+' is '+str(total_loss/total_num_frames)+', num frames is '+str(total_num_frames))

            if error_name == 'train':
                valid_error = epoch_rmsle_loss
                if valid_error < best_validation_loss:
                    early_stop = 0
                    logger.info('saving model, '+nnets_file_name)
                    try:
                        save_path = rnn_model.saver.save(sess, nnets_file_name)
                        # logger.info('use TF saver')
                    except:
                        cPickle.dump(rnn_model, open(nnets_file_name, 'wb'))
                        logger.info('cannot use TF saver; use cPickle')
                    best_validation_loss = valid_error
                elif valid_error > previous_valid_loss:
                    early_stop = early_stop + 1
                    logger.info('reduce learning rate to '+str(rnn_model.learning_rate*0.5))
                    rnn_model.update_learning_rate(rnn_model.learning_rate*0.5)
                if early_stop > early_stop_epoch:
                    early_stop = 0
                    num_roll_back = num_roll_back + 1
                    if num_roll_back > 10:
                        logger.info('reloading ' + str(num_roll_back) + ' times, stopping early, best training '+str(best_validation_loss))
                        return best_validation_loss
                    logger.info('loading previous best model, '+nnets_file_name)
                    try:
                        rnn_model.saver.restore(sess, nnets_file_name)
                        # logger.info('use TF saver')
                    except:
                        rnn_model = cPickle.load(open(nnets_file_name, 'rb'))
                        logger.info('cannot use TF saver; use cPickle')
                    logger.info('reduce learning rate to '+str(rnn_model.learning_rate*0.5))
                    rnn_model.update_learning_rate(rnn_model.learning_rate*0.5)
                previous_valid_loss = valid_error

        epoch_valid_time = time.time()
        output_string = output_string + ', \n  train time is %.2f, test time is %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string)
    sess.close()

if __name__ == '__main__': 
       
    train_NN_iv()