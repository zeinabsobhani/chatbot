from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import pinecone
import os
from data import DataLoader, Preprocessor

class EmbedUpsert:
    def __init__(self, model = 'all-MiniLM-L6-v2', text_col = 'titlebody', metadata_cols = [], batch = 1000):
        self.model = SentenceTransformer(model)
        self.text_col = text_col
        self.metadata_cols = metadata_cols
        self.batch = batch

    def batch_embed(self, df):
        length = max(df.index//self.batch) + 1
        for i,g in tqdm(df.groupby(df.index//self.batch), total=length):
            metadata = g[self.metadata_cols].to_dict(orient='records')
            texts = g[self.text_col].tolist()
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = self.model.encode(texts)
            yield embeds, metadata, ids


    def pinecone_connection(self, PINECONE_API_KEY = None, PINECONE_ENV = None):
        if PINECONE_API_KEY is None:
            PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        if PINECONE_API_KEY is None:
            raise Exception("Please provide PINECONE_API_KEY as an environment variable, or pass in as an argument")
        
        if PINECONE_ENV is None:
            PINECONE_ENV = os.environ.get('PINECONE_ENV')
        if PINECONE_ENV is None:
            raise Exception("Please provide PINECONE_ENV as an environment variable, or pass in as an argument")
        
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

    def batch_upsert(self,df, index_name , namespace = None):
        emb = self.batch_embed(df)
                
        with pinecone.Index(index_name, pool_threads=30) as index:
            # Send requests in parallel
            async_results = [
                index.upsert(vectors=zip(ids, embeds.tolist(), metadatas), async_req=True, namespace=namespace)
                for embeds,metadatas,ids in emb
            ]
            print(async_results)
            # Wait for and retrieve responses (this raises in case of error)
            [async_result.get() for async_result in async_results]


