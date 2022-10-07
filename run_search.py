
from music_search.search import NearestNeighbor
import json


def main():
    DB_PATH = "./db.sqlite"
    DATASET_PATH = "./data/album_metadata.json"
    INDEX_PATH = "./search.index"


    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    nn = NearestNeighbor(INDEX_PATH, DB_PATH)
    nn.build(dataset)
    records = nn.search("The obsessed performing arts student is one of Hollywood’s favorite clichés. Movies like Whiplash, The Perfection, and Nocturne verge on melodrama, detailing the oppressive confines of classical training to varying degrees of absurdity. Their tortured protagonists meet one of two fates: triumph or crack-up. UK duo Jockstrap sound like they are flailing toward both. Graduates of London’s prestigious Guildhall School of Music & Drama, Georgia Ellery and Taylor Skye have made a career of tearing down the academy walls. Their early revolt was scrappy and hardheaded; 2020’s Wicked City EP sounded like two star pupils lashing out, constructing jagged sculptures of string instruments and synthesizers. On their long-awaited debut album, I Love You Jennifer B, they refine their plan of attack. With the help of an 18-piece orchestra, Jockstrap stage elaborate, theatrical scenes atop the conservatory rubble.", k=8)
    [print(r) for r in records]


if __name__ == '__main__':
    main()
