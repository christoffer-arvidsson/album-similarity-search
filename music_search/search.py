from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import music_search.db as db

BATCH_SIZE = 64

def _split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

class NearestNeighbor:
    def __init__(self, index_path: str, db_path: str):
        self._index_path = index_path
        self._db_path = db_path
        self._emb_dim = 768

        self.index = faiss.index_factory(
            self._emb_dim, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT
        )

        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

    @staticmethod
    def transform_dataset(text_dataset):
        sent_doc_ids = []
        sentences = []
        names = []
        for document_id, document_dict in enumerate(text_dataset):
            sent = document_dict["review"].split("\n")
            n_sent = len(sentences)
            sent_doc_ids.extend([document_id] * n_sent)
            sentences.extend(sent)
            names.extend([document_dict["title"]] * n_sent)

        return sentences, sent_doc_ids, names

    def build(self, text_dataset: List[Dict[str, str]]):
        sentences, sent_doc_ids, names = self.transform_dataset(text_dataset)
        doc_ids = list(range(max(sent_doc_ids)))
        sent_ids = np.arange(len(sentences), dtype=int)

        batches = _split_given_size(sentences, BATCH_SIZE)
        embeddings = []
        for batch in tqdm(batches, desc="Embedding batches"):
            embeddings.append(self.embedding_model.encode(batch))

        embeddings = np.concatenate(embeddings, axis=0)

        faiss.normalize_L2(embeddings)

        self.index.add_with_ids(embeddings, sent_ids)

        db_con = db.create_connection(self._db_path)
        db.drop_table(db_con, "sentences")
        db.drop_table(db_con, "documents")
        db.create_tables(db_con)
        db.insert_sentences(db_con, sent_ids.tolist(), sent_doc_ids, sentences)
        db.insert_documents(db_con, doc_ids, names)
        db_con.close()

        
    def search(self, text, k=1):

        sentences = text.split(".")

        embeddings = self.embedding_model.encode(sentences)
        avg_embedding = np.mean(embeddings, axis=0)[None, :]
        faiss.normalize_L2(embeddings)
        distances, indices = self.index.search(avg_embedding, k=k+1)

        # Connect to the database
        db_con = db.create_connection(self._db_path)
        records = db.get_documents_from_sentence_ids(db_con, indices[0].tolist())
        records = list(zip(distances[0], records))
        db_con.close()

        return records

    def write(self):
        faiss.write_index(self.index, self._index_path)



