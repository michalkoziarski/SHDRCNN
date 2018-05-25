import argparse
import data
import model
import os
import json
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from predict import predict
from utils import psnr, tone_mapping, tone_mapping_tf


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-name', type=str, required=True)
parser.add_argument('-n_layers', type=int, default=20)
parser.add_argument('-kernel_size', type=int, default=3)
parser.add_argument('-n_filters', type=int, default=64)
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-weight_decay', type=float, default=0.0005)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-patch_size', type=int, default=41)
parser.add_argument('-stride', type=int, default=21)
parser.add_argument('-n_channels', type=int, default=3)
parser.add_argument('-learning_rate', type=float, default=0.00001)
parser.add_argument('-learning_rate_decay', type=float, default=None)
parser.add_argument('-learning_rate_decay_step', type=int, default=None)
parser.add_argument('-inner_activation', type=str, default='tanh')
parser.add_argument('-outer_activation', type=str, default='sigmoid')
parser.add_argument('-tone_mapping', type=bool, default=True)
parser.add_argument('-discard_well_exposed', type=bool, default=True)
parser.add_argument('-multi_stream', type=bool, default=False)

params = vars(parser.parse_args())

train_set = data.TrainingSet(params['batch_size'], params['patch_size'], params['stride'], params['n_channels'],
                             discard_well_exposed=params['discard_well_exposed'])
validation_set = data.TestSet('ALL', params['n_channels'])

batches_per_epoch = int(np.ceil(train_set.length / train_set.batch_size))

inputs = tf.placeholder(tf.float32)
ground_truth = tf.placeholder(tf.float32)
psnr_t = tf.placeholder(tf.float32, shape=[])
psnr_l = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, trainable=False, name='global_step')

if params['multi_stream']:
    params['n_layers'] = [5, 10, 15, 20, 25, 30]

network = model.Model(inputs, params['n_layers'], params['kernel_size'], params['n_filters'], params['n_channels'],
                      params['inner_activation'], params['outer_activation'])

if params['tone_mapping']:
    base_loss = tf.losses.mean_squared_error(tone_mapping_tf(network.outputs), tone_mapping_tf(ground_truth))
else:
    base_loss = tf.losses.mean_squared_error(network.outputs, ground_truth)

weight_loss = params['weight_decay'] * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in network.weights]))
loss = base_loss + weight_loss

if params['learning_rate_decay'] is None or params['learning_rate_decay_step'] is None:
    learning_rate = params['learning_rate']
else:
    boundaries = [step * batches_per_epoch for step in range(params['learning_rate_decay_step'],
                                                             params['epochs'],
                                                             params['learning_rate_decay_step'])]
    values = [params['learning_rate'] * (params['learning_rate_decay'] ** i) for i in range(len(boundaries) + 1)]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)

tf.summary.scalar('base_loss', base_loss)
tf.summary.scalar('weight_loss', weight_loss)
tf.summary.scalar('total_loss', loss)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('psnr_t', psnr_t)
tf.summary.scalar('psnr_l', psnr_l)
tf.summary.image('inputs', inputs)
tf.summary.image('outputs', network.outputs)
tf.summary.image('ground_truth', ground_truth)
tf.summary.image('residual', network.residual)

for i in range(len(network.weights)):
    tf.summary.histogram('weights/layer_%d' % (i + 1), network.weights[i])
    tf.summary.histogram('biases/layer_%d' % (i + 1), network.biases[i])

summary_step = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=None)

root_models_path = os.path.join(os.path.dirname(__file__), 'models')
root_logs_path = os.path.join(os.path.dirname(__file__), 'logs')
root_params_path = os.path.join(os.path.dirname(__file__), 'params')
root_results_path = os.path.join(os.path.dirname(__file__), 'results')

checkpoint_path = os.path.join(root_models_path, params['name'])
model_path = os.path.join(checkpoint_path, 'model.ckpt')
log_path = os.path.join(root_logs_path, params['name'])

summary_writer = tf.summary.FileWriter(log_path)

for path in [root_models_path, root_logs_path, root_params_path, root_results_path, checkpoint_path, log_path]:
    if not os.path.exists(path):
        os.mkdir(path)

with tf.Session() as session:
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint and checkpoint.model_checkpoint_path:
        logging.info('Restoring model...')

        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        logging.info('Initializing new model...')

        session.run(tf.global_variables_initializer())

        logging.info('Saving parameters...')

        with open(os.path.join(root_params_path, '%s.json' % params['name']), 'w') as f:
            json.dump(params, f)

    logging.info('Training model...')

    batches_processed = tf.train.global_step(session, global_step)
    epochs_processed = int(batches_processed * params['batch_size'] / train_set.length)

    for epoch in range(epochs_processed, params['epochs']):
        train_set.shuffle()

        logging.info('Processing epoch #%d...' % (epoch + 1))

        for batch in tqdm(range(0, batches_per_epoch - 1)):
            x, y = train_set.batch()

            feed_dict = {inputs: x, ground_truth: y}

            session.run([train_step], feed_dict=feed_dict)

        logging.info('Evaluating performance...')

        predictions = predict(validation_set.ldr_images, session, network)

        epoch_psnr_t = np.mean([psnr(tone_mapping(x), tone_mapping(y))
                                for x, y in zip(predictions, validation_set.hdr_images)])
        epoch_psnr_l = np.mean([psnr(x, y)
                                for x, y in zip(predictions, validation_set.hdr_images)])

        x, y = train_set.batch()

        feed_dict = {
            inputs: x,
            ground_truth: y,
            psnr_t: epoch_psnr_t,
            psnr_l: epoch_psnr_l
        }

        _, summary = session.run([train_step, summary_step], feed_dict=feed_dict)

        logging.info('Observed PSNR-T = %.2f and PSNR-L = %.2f.' % (epoch_psnr_t, epoch_psnr_l))

        saver.save(session, model_path, global_step=epoch)
        summary_writer.add_summary(summary, epoch)

    logging.info('Training complete.')

    with open(os.path.join(root_results_path, '%s.json' % params['name']), 'w') as f:
        logging.info('Saving results...')

        json.dump({'psnr-t': epoch_psnr_t, 'psnr-l': epoch_psnr_l}, f)
