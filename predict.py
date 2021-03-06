import utils
import os
import json
import argparse
import imageio
import numpy as np
import tensorflow as tf

from model import Model
from skimage.color import rgb2gray


def load_model(name, session):
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', name)

    assert os.path.exists(checkpoint_path)

    with open(os.path.join(os.path.dirname(__file__), 'params', '%s.json' % name)) as f:
        params = json.load(f)

    inputs = tf.placeholder(tf.float32)
    network = Model(inputs, params['n_layers'], params['kernel_size'], params['n_filters'], params['n_channels'],
                    params['inner_activation'], params['outer_activation'])
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict(images, session=None, model=None, model_name=None, targets=None):
    session_passed = session is not None

    if not session_passed:
        session = tf.Session()

    if model is None:
        model = load_model(model_name, session)

    predictions = []

    if targets is not None:
        psnr = []

    for i in range(len(images)):
        image = images[i].copy()

        assert str(image.dtype).startswith('float') or str(image.dtype).startswith('uint')

        if image.dtype == 'uint8':
            image = image / 255.0
        elif image.dtype == 'uint16':
            image = image / 65535.0

        if len(image.shape) == 2:
            image = np.expand_dims(rgb2gray(image), axis=2)

        prediction = model.outputs.eval(feed_dict={model.inputs: np.array([image])}, session=session)[0]

        if targets is not None:
            target = targets[i].copy()

            assert str(target.dtype).startswith('float') or str(target.dtype).startswith('uint')

            if target.dtype == 'uint8':
                target = target / 255.0
            elif target.dtype == 'uint16':
                target = target / 65535.0

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
    parser.add_argument('-model', help='a name of the trained model that will be used during prediction', required=True)

    args = vars(parser.parse_args())

    if os.path.isfile(args['in']):
        image = imageio.imread(args['in'])
        prediction = predict([image], model_name=args['model'])[0]
        imageio.imwrite(args['out'], prediction)
    elif os.path.isdir(args['in']):
        images = []
        file_names = []

        for file_name in os.listdir(args['in']):
            images.append(imageio.imread(os.path.join(args['in'], file_name)))
            file_names.append(file_name)

        predictions = predict(images, model_name=args['model'])

        if not os.path.exists(args['out']):
            os.mkdir(args['out'])

        for file_name, prediction in zip(file_names, predictions):
            imageio.imsave(os.path.join(args['out'], file_name), prediction)
    else:
        raise ValueError('Incorrect input path.')
