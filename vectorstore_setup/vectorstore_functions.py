
from langchain.vectorstores import Chroma

class ChromaVectorStoreManager:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        self.vectorstore = None

    def create_vector_store(self):
        try:
            self.vectorstore = Chroma.from_documents(documents=self.documents, embedding=self.embedding_model)
        except InvalidDimensionException:
            Chroma().delete_collection()
            self.vectorstore = Chroma.from_documents(documents=self.documents, embedding=self.embedding_model)
    
