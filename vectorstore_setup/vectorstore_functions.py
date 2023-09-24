from langchain.vectorstores import Chroma
from llama_index import VectorStoreIndex, ServiceContext
from chromadb.errors import InvalidDimensionException

class VectorStoreManager:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        self.vectorstore = None
    
    def create_vector_store(self):
        # To be implemented in subclasses
        pass

class ChromaVectorStoreManager(VectorStoreManager):  
    def __init__(self, documents, embedding_model):
        super().__init__(documents, embedding_model)
        self.create_vector_store()
        
    def create_vector_store(self):
        try:
            self.vectorstore = Chroma.from_documents(documents=self.documents, embedding=self.embedding_model)
        except InvalidDimensionException:
            Chroma().delete_collection()
            self.vectorstore = Chroma.from_documents(documents=self.documents, embedding=self.embedding_model)

class VectorStoreIndexManager(VectorStoreManager):
    def __init__(self, documents, embedding_model, service_context):
        super().__init__(documents, embedding_model)
        self.service_context = service_context
        self.create_vector_store()
        
    def create_vector_store(self):
        self.vectorstore = VectorStoreIndex.from_documents(
        self.documents, 
        service_context=self.service_context,
    )