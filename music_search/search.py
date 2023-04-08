from dataclasses import dataclass
from logging import getLogger
from typing import List

import faiss
import numpy as np
from tqdm import tqdm

import music_search.db as db

from transformers import BertTokenizer, BertModel

LOG = getLogger(__name__)

@dataclass
class SearchResult:
    album_metadata: db.AlbumMetadata
    paragraph: str
    score: float


class Embedder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_model = BertModel.from_pretrained("bert-base-uncased")

    def embed(self, texts: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        embeddings = self.embedding_model(**encoded_input)["pooler_output"].detach().numpy()

        return embeddings

class Searcher:
    def __init__(self, index_path: str, db_path: str, batch_size: int = 16, read_index=False):
        self._index_path = index_path
        self._db_path = db_path
        self._emb_dim = 768
        self._batch_size = batch_size

        if read_index:
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.index_factory(
                self._emb_dim, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT
            )

        self.embedder = Embedder()

    def build(self, save: bool = True):
        with db.AlbumDatabase(self._db_path) as db_handler:
            paragraph_ids, paragraphs = db_handler.get_all_reviews()
            paragraph_ids = np.array(paragraph_ids, dtype=int)

        indices = np.arange(paragraph_ids.shape[0])
        batched_indices = np.split(indices, np.arange(self._batch_size, len(indices), self._batch_size))

        for batch_idxs in tqdm(batched_indices, desc="Adding batch"):
            ids = paragraph_ids[batch_idxs]
            paras = [paragraphs[i] for i in batch_idxs]
            embeddings = self.embedder.embed(paras)

            faiss.normalize_L2(embeddings)
            self.index.add_with_ids(embeddings, ids)
        
        if save:
            LOG.info(f"Writing index to {self._index_path}")
            self.write()

    def search(self, text: str, num_neighbors: int=1) -> List[SearchResult]:
        embeddings = self.embedder.embed([text])
        faiss.normalize_L2(embeddings)

        similarities, search_ids = self.index.search(embeddings, k=num_neighbors + 1)

        with db.AlbumDatabase(self._db_path) as db_handler:
            records = db_handler.get_albums_from_paragraph_ids(search_ids[0].tolist())
        search_results = [
            SearchResult(album_metadata=metadata, paragraph=paragraph, score = similarities[0][i])
            for i, (metadata, paragraph) in enumerate(records)
        ]

        return search_results

    def write(self):
        faiss.write_index(self.index, self._index_path)
