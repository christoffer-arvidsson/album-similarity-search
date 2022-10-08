from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk


import music_search.db as db
from logging import getLogger

LOG = getLogger(__name__)

class NearestNeighbor:
    def __init__(self, index_path: str, db_path: str, read_index=False):
        self._index_path = index_path
        self._db_path = db_path
        self._emb_dim = 768

        if read_index:
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.index_factory(
                self._emb_dim, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT
            )

        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

    def sent_tokenize(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)

    def transform_dataset(self, text_dataset):
        elements = []
        for document_id, document_dict in enumerate(text_dataset):
            sent = self.sent_tokenize(document_dict["review"])
            elements.extend([(s, document_id) for s in sent])

        return list(zip(*elements))

    def build(self, text_dataset: List[Dict[str, str]], save: bool=False):
        sentences, sent_doc_ids = self.transform_dataset(text_dataset)
        doc_ids = list(range(len(text_dataset)))
        sent_ids = list(range(len(sentences)))
        doc_titles = [d["title"] for d in text_dataset]

        embeddings = self.embedding_model.encode(sentences)
        faiss.normalize_L2(embeddings)

        self.index.add_with_ids(embeddings, np.array(sent_ids))

        db_con = db.create_connection(self._db_path)
        db.drop_table(db_con, "sentences")
        db.drop_table(db_con, "documents")
        db.create_tables(db_con)
        db.insert_sentences(db_con, sent_ids, sent_doc_ids, sentences)
        db.insert_documents(db_con, doc_ids, doc_titles)
        db_con.close()

        if save:
            LOG.info(f"Writing index to {self._index_path}")
            self.write()
        
    def search(self, text, k=1):
        sentences = self.sent_tokenize(text)

        embeddings = self.embedding_model.encode(sentences)
        faiss.normalize_L2(embeddings)
        embeddings = np.mean(embeddings, axis=0)[None, :]
        distances, indices = self.index.search(embeddings, k=k+1)

        # Connect to the database
        db_con = db.create_connection(self._db_path)
        records = db.get_documents_from_sentence_ids(db_con, indices[0].tolist())
        records = list(zip(distances[0], records))
        db_con.close()

        return records

    def write(self):
        faiss.write_index(self.index, self._index_path)



