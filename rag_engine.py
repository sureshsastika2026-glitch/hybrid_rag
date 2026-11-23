import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

COLLECTION_NAME = "spotify_architecture"

class HybridRAGEngine:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 1. Initialize Qdrant (Vector DB)
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
        if not self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )

        # 2. Initialize Neo4j (Knowledge Graph)
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

    def ingest_pdf(self, pdf_path: str):
        """Loads PDF, chunks it, indexes into Qdrant, and builds Neo4j Knowledge Graph."""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        # A. Store in Vector DB (Qdrant)
        self.vector_store.add_documents(chunks)

        # B. Extract Entities & Relationships for Knowledge Graph (Neo4j)
        # Tailored schema for software architecture documents
        graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[
                "Microservice", "Technology", "Database", "Table", "Route", "Feature"
            ],
            allowed_relationships=[
                "USES", "STORES_IN", "CONTAINS_TABLE", "COMMUNICATES_WITH", "IMPLEMENTS"
            ],
        )
        graph_documents = graph_transformer.convert_to_graph_documents(chunks)
        self.graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        return len(chunks), len(graph_documents)

    def _query_graph(self, question: str) -> str:
        """Extracts key entities from the question and fetches 1-hop neighborhoods from Neo4j."""
        entity_prompt = ChatPromptTemplate.from_template(
            "Extract up to 3 key technical entity names (e.g., PostgreSQL, Auth Service, Next.js) "
            "from the question as a comma-separated list. Return ONLY the names.\nQuestion: {question}"
        )
        raw_entities = (entity_prompt | self.llm).invoke({"question": question}).content
        entities = [e.strip() for e in raw_entities.split(",") if e.strip()]

        graph_context = []
        for entity in entities:
            cypher = """
            MATCH (n)-[r]-(m)
            WHERE toLower(n.id) CONTAINS toLower($entity)
            RETURN n.id AS source, type(r) AS rel, m.id AS target
            LIMIT 15
            """
            results = self.graph.query(cypher, params={"entity": entity})
            for row in results:
                graph_context.append(f"({row['source']}) -[{row['rel']}]-> ({row['target']})")

        return "\n".join(set(graph_context)) if graph_context else "No direct graph relationships found."

    def query(self, question: str):
        """Runs Hybrid Retrieval (Vector + Graph) and synthesizes the final response."""
        # 1. Vector Retrieval
        vector_docs = self.vector_store.similarity_search(question, k=4)
        vector_context = "\n\n".join(
            [f"[Page {d.metadata.get('page', 0) + 1}]: {d.page_content}" for d in vector_docs]
        )

        # 2. Knowledge Graph Retrieval
        graph_context = self._query_graph(question)

        # 3. Hybrid Synthesis
        prompt = ChatPromptTemplate.from_template(
            """You are an expert Software Architect answering questions about a Spotify-like web app architecture.
            Combine both the Vector Context (detailed text) and Knowledge Graph Context (entity relationships) to answer accurately.

            --- VECTOR CONTEXT ---
            {vector_context}

            --- KNOWLEDGE GRAPH CONTEXT ---
            {graph_context}

            Question: {question}
            Answer:"""
        )
        chain = prompt | self.llm
        answer = chain.invoke({
            "vector_context": vector_context,
            "graph_context": graph_context,
            "question": question,
        }).content

        return {
            "answer": answer,
            "vector_context": vector_context,
            "graph_context": graph_context,
        }