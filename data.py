import os
import zipfile
import logging
import imageio
import numpy as np

from tqdm import tqdm
from skimage.color import rgb2gray
from urllib.request import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TrainingSet:
    N_IMAGES = 74
    N_AUGMENTATIONS = 8

    def __init__(self, batch_size=64, patch_size=41, stride=21, n_channels=3, shape=(1000, 1500)):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.n_channels = n_channels
        self.shape = shape
        self.images_completed = 0
        self.epochs_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'Training')

        if not os.path.exists(self.root_path):
            download('Training')

        logging.info('Loading training partition...')

        self.length = 0

        x_start = 0

        while x_start + patch_size <= shape[0]:
            y_start = 0

            while y_start + patch_size <= shape[1]:
                self.length += 1

                y_start += patch_size - stride

            x_start += patch_size - stride

        self.length *= self.N_IMAGES * self.N_AUGMENTATIONS

        self.images = np.empty((self.length, 2, self.patch_size, self.patch_size, self.n_channels), dtype=np.float32)

        current_image = 0

        for i in tqdm(range(1, self.N_IMAGES + 1)):
            image_directory = os.path.join(self.root_path, '%03d' % i)

            ldr_image, hdr_image = _load_images(image_directory, self.n_channels, self.shape)

            x_start = 0

            while x_start + patch_size <= shape[0]:
                x_end = x_start + patch_size
                y_start = 0

                while y_start + patch_size <= shape[1]:
                    y_end = y_start + patch_size

                    ldr_patch = ldr_image[x_start:x_end, y_start:y_end].copy()
                    hdr_patch = hdr_image[x_start:x_end, y_start:y_end].copy()

                    for _ in range(4):
                        ldr_patch = np.rot90(ldr_patch)
                        hdr_patch = np.rot90(hdr_patch)

                        self.images[current_image, 0] = ldr_patch
                        self.images[current_image, 1] = hdr_patch

                        current_image += 1

                    ldr_patch = np.fliplr(ldr_patch)
                    hdr_patch = np.fliplr(hdr_patch)

                    for _ in range(4):
                        ldr_patch = np.rot90(ldr_patch)
                        hdr_patch = np.rot90(hdr_patch)

                        self.images[current_image, 0] = ldr_patch
                        self.images[current_image, 1] = hdr_patch

                        current_image += 1

                    y_start += patch_size - stride

                x_start += patch_size - stride

    def batch(self):
        ldr_images = self.images[self.images_completed:(self.images_completed + self.batch_size), 0]
        hdr_images = self.images[self.images_completed:(self.images_completed + self.batch_size), 1]

        self.images_completed += self.batch_size

        if self.images_completed >= self.length:
            self.images_completed = 0
            self.epochs_completed += 1

        return ldr_images, hdr_images

    def shuffle(self):
        logging.info('Shuffling training partition...')

        np.random.shuffle(self.images)

        self.images_completed = 0


class TestSet:
    def __init__(self, portion='PAPER', n_channels=3, shape=(1000, 1500)):
        assert portion in ['PAPER', 'EXTRA']

        self.portion = portion
        self.n_channels = n_channels
        self.shape = shape
        self.images_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'Test', self.portion)

        if not os.path.exists(self.root_path):
            download('Training')

        logging.info('Loading "%s" portion of the test partition...' % self.portion.lower())

        image_directories = os.listdir(self.root_path)

        self.length = len(image_directories)
        self.images = np.empty((self.length, 2, self.shape[0], self.shape[1], self.n_channels), dtype=np.float32)

        for i in tqdm(range(len(image_directories))):
            image_directory = os.path.join(self.root_path, image_directories[i])
            ldr_image, hdr_image = _load_images(image_directory, self.n_channels, self.shape)

            self.images[i][0] = ldr_image
            self.images[i][1] = hdr_image

    def fetch(self):
        ldr_image = self.images[self.images_completed, 0]
        hdr_image = self.images[self.images_completed, 1]

        self.images_completed += 1

        if self.images_completed >= self.length:
            self.images_completed = 0

        return ldr_image, hdr_image


def download(partition):
    assert partition in ['Training', 'Test']

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    zip_path = os.path.join(DATA_PATH, 'SIGGRAPH17_HDR_%sset.zip' % partition)
    url = 'http://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_%sset.zip' % partition

    if not os.path.exists(zip_path):
        logging.info('Downloading %s partition...' % partition.lower())

        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as f:
        logging.info('Extracting %s partition...' % partition.lower())

        f.extractall(DATA_PATH)


def _load_images(image_directory, n_channels=3, shape=(1000, 1500)):
    tif_names = sorted([name for name in os.listdir(image_directory) if name.endswith('.tif')])

    assert len(tif_names) == 3

    ldr_path = os.path.join(image_directory, tif_names[1])
    hdr_path = os.path.join(image_directory, 'HDRImg.hdr')
    ldr_image = imageio.imread(ldr_path)
    hdr_image = imageio.imread(hdr_path)

    assert ldr_image.dtype == 'uint16'
    assert hdr_image.dtype == 'float32'
    assert ldr_image.shape[0] == hdr_image.shape[0] == shape[0]
    assert ldr_image.shape[1] == hdr_image.shape[1] == shape[1]

    ldr_image = ldr_image / 65535.0

    if n_channels == 1:
        ldr_image = np.expand_dims(rgb2gray(ldr_image), axis=2)
        hdr_image = np.expand_dims(rgb2gray(hdr_image), axis=2)

    return ldr_image, hdr_image


if __name__ == '__main__':
    for partition in ['Training', 'Test']:
        download(partition)
