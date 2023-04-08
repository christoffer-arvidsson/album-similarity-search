
import logging

from music_search.search import Searcher
from music_search.util import get_config, read_dataset

LOG = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    config = get_config("./configs/default.yaml")

    LOG.info("Building index")
    nn = Searcher(config.index_path, config.db_path)
    nn.build()

if __name__ == '__main__':
    main()
