
from music_search.search import NearestNeighbor
import logging

from music_search.util import get_config, read_dataset

LOG = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    config = get_config("./configs/default.yaml")

    LOG.info("Loading dataset")
    dataset = read_dataset(config.dataset_path)

    LOG.info("Building index")
    nn = NearestNeighbor(config.index_path, config.db_path)
    nn.build(dataset, save=True)

if __name__ == '__main__':
    main()
