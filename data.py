import os
import zipfile

from urllib.request import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TrainingSet:
    def __init__(self):
        pass


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
