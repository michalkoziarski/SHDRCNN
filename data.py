import os
import zipfile
import imageio
import numpy as np

from skimage.color import rgb2gray
from urllib.request import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TrainingSet:
    def __init__(self, batch_size=64, patch_size=41, stride=21):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.images_completed = 0
        self.epochs_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'Training')
        self.ldr_images = []
        self.hdr_images = []

        if not os.path.exists(self.root_path):
            download('Training')

        for i in range(1, 75):
            image_directory = os.path.join(self.root_path, '%03d' % i)
            tif_names = sorted([name for name in os.listdir(image_directory) if name.endswith('.tif')])

            assert len(tif_names) == 3

            ldr_path = os.path.join(image_directory, tif_names[1])
            hdr_path = os.path.join(image_directory, 'HDRImg.hdr')
            ldr_image = imageio.imread(ldr_path)
            hdr_image = imageio.imread(hdr_path)

            assert ldr_image.dtype == 'uint16'
            assert hdr_image.dtype == 'float32'
            assert ldr_image.shape == hdr_image.shape

            ldr_image = ldr_image / 65535.0

            ldr_image = np.expand_dims(rgb2gray(ldr_image), axis=2)
            hdr_image = np.expand_dims(rgb2gray(hdr_image), axis=2)

            x_start = 0

            while x_start + patch_size <= ldr_image.shape[0]:
                x_end = x_start + patch_size
                y_start = 0

                while y_start + patch_size <= ldr_image.shape[1]:
                    y_end = y_start + patch_size

                    ldr_patch = ldr_image[x_start:x_end, y_start:y_end].copy()
                    hdr_patch = hdr_image[x_start:x_end, y_start:y_end].copy()

                    for _ in range(4):
                        ldr_patch = np.rot90(ldr_patch)
                        hdr_patch = np.rot90(hdr_patch)

                        self.ldr_images.append(ldr_patch)
                        self.hdr_images.append(hdr_patch)

                    ldr_patch = np.fliplr(ldr_patch)
                    hdr_patch = np.fliplr(hdr_patch)

                    for _ in range(4):
                        ldr_patch = np.rot90(ldr_patch)
                        hdr_patch = np.rot90(hdr_patch)

                        self.ldr_images.append(ldr_patch)
                        self.hdr_images.append(hdr_patch)

                    y_start += patch_size - stride

                x_start += patch_size - stride

        self.ldr_images = np.array(self.ldr_images)
        self.hdr_images = np.array(self.hdr_images)

        self.shuffle()
        self.length = len(self.ldr_images)
        self.length = self.length - self.length % batch_size
        self.ldr_images = self.ldr_images[:self.length]
        self.hdr_images = self.hdr_images[:self.length]

    def batch(self):
        ldr_images = self.ldr_images[self.images_completed:(self.images_completed + self.batch_size)]
        hdr_images = self.hdr_images[self.images_completed:(self.images_completed + self.batch_size)]

        self.images_completed += self.batch_size

        if self.images_completed >= self.length:
            self.images_completed = 0
            self.epochs_completed += 1
            self.shuffle()

        return ldr_images, hdr_images

    def shuffle(self):
        indices = list(range(len(self.ldr_images)))
        np.random.shuffle(indices)

        self.ldr_images = self.ldr_images[indices]
        self.hdr_images = self.hdr_images[indices]


class TestSet:
    def __init__(self):
        pass


def download(partition):
    assert partition in ['Training', 'Test']

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    zip_path = os.path.join(DATA_PATH, 'SIGGRAPH17_HDR_%sset.zip' % partition)
    url = 'http://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_%sset.zip' % partition

    if not os.path.exists(zip_path):
        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as f:
        f.extractall(DATA_PATH)


if __name__ == '__main__':
    for partition in ['Training', 'Test']:
        download(partition)
