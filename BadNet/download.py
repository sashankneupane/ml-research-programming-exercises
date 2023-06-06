import pathlib
from dataset.poison import build_init_data
from utils import dotdict

download_args = {
    'dataset': 'cifar10',
    'data_path': './dataset/cifar10',
}
download_args = dotdict(download_args)


# download dataset
def download_dataset(args):
    data_path = args.data_path
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    build_init_data(args)

download_dataset(download_args)