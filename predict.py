import model
import utils
import os
import json
import argparse
import imageio
import numpy as np
import tensorflow as tf

from skimage.color import rgb2gray


def load_model(session):
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'model')

    assert os.path.exists(checkpoint_path)

    with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
        params = json.load(f)

    inputs = tf.placeholder(tf.float32)
    network = model.Model(inputs, params['n_layers'], params['kernel_size'], params['n_filters'])
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict(images, session=None, network=None, targets=None):
    session_passed = session is not None

    if not session_passed:
        session = tf.Session()

    if network is None:
        network = load_model(session)

    predictions = []

    if targets is not None:
        psnr = []

    for i in range(len(images)):
        image = images[i].copy()

        assert image.dtype == 'float32'

        if len(image.shape) == 3:
            image = np.expand_dims(rgb2gray(image), axis=2)

        prediction = network.outputs.eval(feed_dict={network.inputs: np.array([image])}, session=session)[0]

        if targets is not None:
            target = targets[i]

            assert target.dtype == 'float32'

            psnr.append(utils.psnr(prediction, target, maximum=1.0))

        predictions.append(prediction)

    if not session_passed:
        session.close()

    if targets is not None:
        return predictions, psnr
    else:
        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', help='a path of the input image or a directory of the input images', required=True)
    parser.add_argument('-out', help='a path for the output image or a directory for the output images', required=True)
    args = vars(parser.parse_args())

    if os.path.isfile(args['in']):
        image = imageio.imread(args['in'])
        prediction = predict([image])[0]
        imageio.imwrite(args['out'], prediction)
    elif os.path.isdir(args['in']):
        images = []
        file_names = []

        for file_name in os.listdir(args['in']):
            images.append(imageio.imread(os.path.join(args['in'], file_name)))
            file_names.append(file_name)

        predictions = predict(images)

        if not os.path.exists(args['out']):
            os.mkdir(args['out'])

        for file_name, prediction in zip(file_names, predictions):
            imageio.imsave(os.path.join(args['out'], file_name), prediction)
    else:
        raise ValueError('Incorrect input path.')
