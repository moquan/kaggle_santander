import os, sys, cPickle, time, shutil, logging
sys.stdout.flush()
import math, numpy, scipy, random
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.lstm_ops import LSTMBlockFusedCell
from modules import make_logger
import cPickle
# import numpy as np
logger = make_logger("Main")

work_dir = "/home/dawna/tts/mw545/DVExp/hh/"
scratch_dir = "/scratch/tmp-mw545/hh/"
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

min_max_file_name=work_dir+"min_max.dat"
logger.info('loading min max file '+min_max_file_name)
min_max = cPickle.load(open(min_max_file_name, 'rb'))
# y_diff = (min_max["y_max"] - min_max["y_min"])

class acoustic_model_cfg(object):
    def __init__(self, work_dir="/home/dawna/tts/mw545/DVExp/hh"):

        # self.input_dim  = 21
        self.input_dim  = 9470 

        self.batch_size = 4459
        self.learning_rate = 0.0001
        self.num_epoch     = 200000
        self.early_stop_epoch = 5
        # self.rnn_layer_type = ['LSTM'] * 4
        self.rnn_layer_type = ['Relu'] * 10
        # self.rnn_layer_type = ['TANH', 'LSTM', 'LSTM', 'TANH', 'TANH']
        # self.high_way = [False, False, False, True, True]
        # self.high_way = [False] + [True] * 29
        self.high_way = [False] * 10
        # self.rnn_layer_size = [ 128 ] * 5
        self.rnn_layer_size = [ 256 ] * 10
        self.dropout_prob   = [ 0.5 ] * 10
        self.num_rnn_layers = len(self.rnn_layer_type)
        assert self.num_rnn_layers == len(self.rnn_layer_size)
        self.gpu_id = 0
        self.gpu_per_process_gpu_memory_fraction = 0.8

        self.output_dim = 1

        exp_dir = "exp_dnn_"+str(self.num_rnn_layers)
        for i in range(min(self.num_rnn_layers, 5)):
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
            self.DNNGEN    = True

class build_acoustic_model(object):

    def __init__(self, amcfg):

        with tf.device('/device:GPU:'+str(amcfg.gpu_id)):
            self.am_input    = tf.placeholder(tf.float64, shape=[amcfg.batch_size, amcfg.input_dim])
            self.am_target   = tf.placeholder(tf.float64, shape=[amcfg.batch_size, amcfg.output_dim])

            self.output_list = []
            self.rnn_layers = {}
            self.rnn_layer_output  = {}
            self.rnn_layer_c_h = {}
            self.rnn_init_c    = {}
            self.rnn_init_h    = {}
            self.learning_rate = amcfg.learning_rate
            self.learning_rate_holder = tf.placeholder(dtype=tf.float64, name='acoustic_learning_rate_holder')
            self.dropout_keep_prob = []
            for i in range(amcfg.num_rnn_layers):
                self.dropout_keep_prob.append(tf.placeholder(dtype=tf.float32, name='dropout_keep_prob_'+str(i)))

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

                    if amcfg.dropout_prob[i] > 0:
                        self.rnn_layer_output[i] = tf.nn.dropout(self.rnn_layer_output[i], keep_prob=self.dropout_keep_prob[i], seed=random.randint(0, 545))
                        
                # rnn_layer_output.append(layer_output_temp)
                # rnn_layer_state.append(layer_state_temp)
            with tf.variable_scope('acoustic_final_layer'):
                layer_input = self.rnn_layer_output[amcfg.num_rnn_layers-1]
                self.final_layer_output = tf.contrib.layers.fully_connected(inputs=layer_input, num_outputs=amcfg.output_dim, activation_fn=tf.nn.relu)

                # self.zero_output = tf.zeros(shape=[amcfg.batch_size, amcfg.output_dim])
                # self.denorm_labels  = (self.am_target 
                    # + tf.constant(1., dtype=tf.float64)) * tf.constant(y_diff/2., dtype=tf.float64) + tf.constant(min_max["y_min"], dtype=tf.float64)
                # self.denorm_predict = tf.maximum((self.final_layer_output 
                    # + tf.constant(1.)) * tf.constant(y_diff/2.) + tf.constant(min_max["y_min"]), self.zero_output)
                # self.denorm_predict = (self.final_layer_output 
                    # + tf.constant(1., dtype=tf.float64)) * tf.constant(y_diff/2., dtype=tf.float64) + tf.constant(min_max["y_min"], dtype=tf.float64)

                self.denorm_labels   = self.am_target * tf.constant(min_max["y_max"]/2., dtype=tf.float64)
                self.denorm_predict  = self.final_layer_output * tf.constant(min_max["y_max"]/2., dtype=tf.float64)

                self.log_labels  = tf.log( self.denorm_labels  + tf.constant(1., dtype=tf.float64))
                self.log_predict = tf.log( self.denorm_predict + tf.constant(1., dtype=tf.float64))

                self.loss = tf.losses.mean_squared_error(labels=self.log_labels, predictions=self.log_predict)
                self.mse  = tf.losses.mean_squared_error(labels=self.am_target, predictions=self.final_layer_output)

                # self.loss  = tf.losses.mean_squared_error(labels=self.am_target, predictions=self.final_layer_output)
                # self.accuracy = tf.count_nonzero(tf.greater(self.am_target * self.final_layer_output, 0.)) * tf.constant(float(1./amcfg.batch_size), name="1_/batch_size")

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

def make_a_batch_gen(amcfg, data_x, batch_idx, batch_size):
    x = numpy.zeros((amcfg.batch_size, amcfg.input_dim))
    y = numpy.zeros((amcfg.batch_size, amcfg.output_dim))

    T = data_x.shape[0]
    start_idx = batch_idx * batch_size
    end_idx   = min((batch_idx+1) * batch_size, T) # Actually this is N in 0:N, so last index is N-1
    data_len  = end_idx - start_idx

    x[:data_len] = data_x[start_idx:end_idx]

    return x, y

def train_NN():

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

    data_norm_file_name=scratch_dir+"data_norm.dat"
    try:
        logger.info('loading normalised data file '+data_norm_file_name)
        data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
    except:
        logger.info('copy to scratch normalised data file '+data_norm_file_name)
        shutil.copyfile(work_dir+"data_norm.dat", data_norm_file_name)
        logger.info('loading normalised data file '+data_norm_file_name)
        data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
    logger.info(data_norm["x_train"].shape)
    logger.info(data_norm["y_train"].shape)

    early_stop = 0
    epoch = 0
    num_roll_back = 0
    best_validation_loss = sys.float_info.max
    previous_valid_loss  = sys.float_info.max
    num_batch = {
        "train": 5,
        # "test":  int(44812/amcfg.batch_size)
    }

    training_epochs  = amcfg.num_epoch
    early_stop_epoch = amcfg.early_stop_epoch
    nnets_file_name = amcfg.nnets_file_name
    
    while (epoch < training_epochs):
        
        epoch_start_time = time.time()

        if epoch > 0:
            logger.info('start training Epoch '+str(epoch))
            
            for batch_idx in range(num_batch["train"]):
                x, y = make_a_batch(amcfg, data_norm["x_train"], data_norm["y_train"], amcfg.batch_size)
                feed_dict = {}
                feed_dict[rnn_model.am_input]  = x
                feed_dict[rnn_model.am_target] = y
                feed_dict[rnn_model.learning_rate_holder] = rnn_model.learning_rate
                for i in range(amcfg.num_rnn_layers):
                    feed_dict[rnn_model.dropout_keep_prob[i]] = amcfg.dropout_prob[i]
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
                for i in range(amcfg.num_rnn_layers):
                    feed_dict[rnn_model.dropout_keep_prob[i]] = 0.
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
                        rnn_model.saver.save(sess, nnets_file_name)
                        # logger.info('use TF saver')
                    except:
                        cPickle.dump(rnn_model, open(nnets_file_name, 'wb'))
                        logger.info('cannot use TF saver; use cPickle')
                    best_validation_loss = valid_error
                elif valid_error >= previous_valid_loss:
                    early_stop = early_stop + 1
                    logger.info('reduce learning rate to '+str(rnn_model.learning_rate*0.5))
                    rnn_model.update_learning_rate(rnn_model.learning_rate*0.5)
                if early_stop > early_stop_epoch:
                    early_stop = 0
                    num_roll_back = num_roll_back + 1
                    # if num_roll_back > 10:
                    #     logger.info('reloading ' + str(num_roll_back) + ' times, stopping early, best training '+str(best_validation_loss))
                    #     return best_validation_loss
                    logger.info('reloading ' + str(num_roll_back) + ' times, loading previous best model, '+nnets_file_name)
                    rnn_model.saver.restore(sess, nnets_file_name)

                previous_valid_loss = valid_error

        epoch_valid_time = time.time()
        output_string = output_string + ', \n  train time is %.2f, test time is %.2f' %((epoch_train_time - epoch_start_time), (epoch_valid_time - epoch_train_time))
        logger.info(output_string)

        epoch = epoch + 1
    sess.close()

def gen_NN():

    test_ID_file_name=work_dir+"test_ID.dat"
    IDs = cPickle.load(open(test_ID_file_name, 'rb'))
    # IDs = prices_raw['ID']

    amcfg = acoustic_model_cfg()
    # amcfg.iv_dim = iv_size
    logger.info('config tensorflow')
    tf_config = config_tf(amcfg)
    logger.info('building model')
    rnn_model = build_acoustic_model(amcfg)
    logger.info('start running model')
    sess = tf.Session(config=tf_config)
    # tf.global_variables_initializer().run()    
    # sess.run(rnn_model.init)
    # rnn_model.saver.restore(sess, nnets_file_name)

    data_norm_file_name=scratch_dir+"data_norm_test.dat"
    try:
        logger.info('loading normalised data file '+data_norm_file_name)
        data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
    except:
        logger.info('copy to scratch normalised data file '+data_norm_file_name)
        shutil.copyfile(work_dir+"data_norm_test.dat", data_norm_file_name)
        logger.info('loading normalised data file '+data_norm_file_name)
        data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
    logger.info(data_norm["x_test"].shape)

    nnets_file_name = amcfg.nnets_file_name
    rnn_model.saver.restore(sess, nnets_file_name)

    num_samples = data_norm["x_test"].shape[0]
    num_batches = int( (num_samples - 1 ) / amcfg.batch_size) + 1
    predict_output_list = []

    for batch_idx in range(num_batches):
        logger.info('start generating')
        x, y = make_a_batch_gen(amcfg, data_norm["x_test"], batch_idx, amcfg.batch_size)
        feed_dict = {}
        feed_dict[rnn_model.am_input]  = x
        feed_dict[rnn_model.am_target] = y
        feed_dict[rnn_model.learning_rate_holder] = rnn_model.learning_rate
        for i in range(amcfg.num_rnn_layers):
            feed_dict[rnn_model.dropout_keep_prob[i]] = 0.
        batch_predict_output = sess.run(fetches=rnn_model.denorm_predict, feed_dict=feed_dict)
        predict_output_list.append(batch_predict_output)

    predict_output = numpy.concatenate(predict_output_list, axis=0)
    T = data_norm["x_test"].shape[0]
    predict_output = predict_output[:T]
    predict_output_file_name = work_dir+"predict_output.dat"
    cPickle.dump(predict_output, open(predict_output_file_name, 'wb'))

    sess.close()

def make_submission():
    import warnings, copy
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set(style="whitegrid", color_codes=True)
    import cPickle

    work_dir = '/home/dawna/tts/mw545/DVExp/hh/'
    work_dir = '/home/dawna/tts/mw545/DVExp/hh/'
    sub_csv  = work_dir + 'predict_submission.csv'

    test_ID_file_name=work_dir+"test_ID.dat"
    IDs = cPickle.load(open(test_ID_file_name, 'rb'))
    # IDs = prices_raw['ID']

    predict_output_file_name = work_dir+"predict_output.dat"
    predict_output = cPickle.load(open(predict_output_file_name, 'rb'))

    predict_output_temp = cPickle.load(open(predict_output_file_name, 'rb'))
    n = predict_output_temp.shape[0] * predict_output_temp.shape[1]
    predict_output = np.zeros((n, 1))
    for i in range(predict_output_temp.shape[1]):
      # print predict_output[i*predict_output_temp.shape[0]:(i+1)*predict_output_temp.shape[0]].shape
      # print predict_output_temp[:,i].shape
      predict_output[i*predict_output_temp.shape[0]:(i+1)*predict_output_temp.shape[0],0] = np.maximum(predict_output_temp[:,i], 0)
    predict_output = predict_output[:49342]
    # T = 49342
    # predict_output = np.zeros((T,1))
    # for i in range(T):
    #   predict_output[i, 0] = i+1

    assert IDs.values.shape[0] == predict_output.shape[0]

    df = pd.DataFrame()
    df['ID'] = IDs
    df['target'] = predict_output[:,0]
    T = ['ID', 'target']
    df = df[T]
    df.to_csv(sub_csv, encoding='utf-8', index=False)


if __name__ == '__main__': 
       
    amcfg = acoustic_model_cfg()
    if amcfg.TRAINDNN:
        train_NN()
    if amcfg.DNNGEN:
        gen_NN()
        make_submission()