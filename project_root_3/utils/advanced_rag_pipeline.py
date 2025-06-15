import os
import json
import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Настройка логирования
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Класс для представления документа"""
    id: str
    content: str
    metadata: Dict = None
    timestamp: str = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class QueryResult:
    """Класс для результатов поиска"""
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict
    embedding: List[float] = None

class DocumentProcessor:
    """Класс для предобработки документов"""

    @staticmethod
    def clean_text(text: str) -> str:
        # Можно подключить rag_text_utils.clean_html, strip_non_printable и т.д.
        return ' '.join(text.strip().split())

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, max_chunks: Optional[int] = None) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunk_size = max(chunk_size, 1)
        overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
        step = max(chunk_size - overlap, 1)
        chunks = []
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if max_chunks and len(chunks) >= max_chunks:
                break
        return chunks

    @classmethod
    def process_document(cls, content: str, chunk_size: int = 500, overlap: int = 50, max_chunks: Optional[int] = None) -> List[str]:
        cleaned = cls.clean_text(content)
        return cls.chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)

class EmbeddingManager:
    """Управление эмбеддингами"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        try:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.model.encode(texts, **kwargs)
            return np.asarray(embeddings)
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            return float(cosine_similarity(embedding1, embedding2)[0][0])
        except Exception as e:
            logger.error(f"Ошибка вычисления similarity: {e}")
            return 0.0

class AdvancedRAGPipeline:
    """Продвинутый RAG pipeline"""

    def __init__(
        self,
        collection_name: str = "advanced_rag",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Инициализация ChromaDB
        self._init_chromadb()

        # Инициализация компонентов
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.processor = DocumentProcessor()

        logger.info("RAG Pipeline initialized successfully")

    def _init_chromadb(self):
        """Инициализация ChromaDB"""
        settings = Settings()
        if self.persist_directory:
            settings.persist_directory = self.persist_directory
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.Client(settings)

        # Создание или получение коллекции
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Advanced RAG Pipeline Collection"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        chunk_size: int = 500,
        overlap: int = 50,
        max_chunks: int = 1000,
        max_doc_len: int = 100_000
    ) -> List[str]:
        if not content or not isinstance(content, str):
            logger.warning("Пустой или невалидный контент, документ не будет добавлен")
            return []
        if len(content) > max_doc_len:
            logger.warning(f"Документ слишком большой ({len(content)}), будет усечён до {max_doc_len}")
            content = content[:max_doc_len]

        # Обработка документа
        chunks = self.processor.process_document(content, chunk_size)

        # Создание эмбеддингов
        embeddings = self.embedding_manager.encode(chunks)

        # Подготовка данных для ChromaDB
        chunk_ids = []
        chunk_documents = []
        chunk_embeddings = []
        chunk_metadata = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_documents.append(chunk)
            chunk_embeddings.append(embedding.tolist())

            chunk_meta = {
                **(metadata if metadata else {}),
                "parent_doc_id": doc_id,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            chunk_metadata.append(chunk_meta)

        # Добавление в коллекцию
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            embeddings=chunk_embeddings,
            metadatas=chunk_metadata
        )

        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        return chunk_ids

    def add_documents_batch(
        self,
        documents: List[Document],
        chunk_size: int = 500,
        batch_size: int = 100
    ) -> List[str]:
        """Пакетное добавление документов"""
        all_chunk_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = []

            for doc in batch:
                chunk_ids = self.add_document(
                    content=doc.content,
                    doc_id=doc.id,
                    metadata=doc.metadata,
                    chunk_size=chunk_size
                )
                batch_ids.extend(chunk_ids)

            all_chunk_ids.extend(batch_ids)
            logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} documents")

        return all_chunk_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
        min_similarity: float = 0.0
    ) -> List[QueryResult]:
        """Поиск документов"""
        # Создание эмбеддинга запроса
        query_embedding = self.embedding_manager.encode([query])[0]

        # Поиск в ChromaDB
        where_clause = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_clause
        )

        # Обработка результатов
        query_results = []

        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else None

            # Преобразование distance в similarity score
            similarity_score = 1 - distance if distance is not None else 0.0

            if similarity_score >= min_similarity:
                query_result = QueryResult(
                    document_id=doc_id,
                    content=content,
                    similarity_score=similarity_score,
                    metadata=metadata
                )
                query_results.append(query_result)

        logger.info(f"Found {len(query_results)} results for query: '{query[:50]}...'")
        return query_results

    def delete_document(self, doc_id: str) -> bool:
        """Удаление документа и всех его чанков"""
        try:
            # Находим все чанки документа
            results = self.collection.get(
                where={"parent_doc_id": doc_id}
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted document {doc_id} and {len(results['ids'])} chunks")
                return True
            else:
                logger.warning(f"Document {doc_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def update_document(
        self,
        doc_id: str,
        new_content: str,
        new_metadata: Optional[Dict] = None,
        chunk_size: int = 500
    ) -> List[str]:
        """Обновление документа"""
        # Удаляем старый документ
        self.delete_document(doc_id)

        # Добавляем новый
        return self.add_document(
            content=new_content,
            doc_id=doc_id,
            metadata=new_metadata,
            chunk_size=chunk_size
        )

    def get_collection_stats(self) -> Dict:
        """Статистика коллекции"""
        try:
            count = self.collection.count()

            # Получаем примеры метаданных
            sample = self.collection.peek(limit=10)

            stats = {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "sample_metadata": sample.get('metadatas', [])[:3] if sample else []
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def export_collection(self, filepath: str, chunk_limit: int = 100_000) -> bool:
        """Экспорт коллекции в JSON"""
        try:
            all_data = self.collection.get(limit=chunk_limit)

            export_data = {
                "collection_name": self.collection_name,
                "export_timestamp": datetime.now().isoformat(),
                "data": {
                    "ids": all_data.get('ids', []),
                    "documents": all_data.get('documents', []),
                    "metadatas": all_data.get('metadatas', []),
                    "embeddings": all_data.get('embeddings', [])
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Collection exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return False

    def semantic_search_with_reranking(
        self,
        query: str,
        n_results: int = 10,
        rerank_top_k: int = 5
    ) -> List[QueryResult]:
        initial_results = self.search(query, n_results=n_results)
        if not initial_results:
            return []
        query_embedding = self.embedding_manager.encode([query])[0]
        # Кэшируем эмбеддинги документов (чтобы не считать повторно)
        doc_texts = [r.content for r in initial_results]
        doc_embeddings = self.embedding_manager.encode(doc_texts)
        reranked_results = []
        for i, result in enumerate(initial_results):
            similarity = self.embedding_manager.compute_similarity(query_embedding, doc_embeddings[i])
            result.similarity_score = similarity
            reranked_results.append(result)
        reranked_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return reranked_results[:rerank_top_k]

# Пример использования
if __name__ == "__main__":
    # Инициализация RAG pipeline
    rag = AdvancedRAGPipeline(
        collection_name="advanced_rag_demo",
        persist_directory="./rag_storage"
    )

    # Тестовые документы
    test_documents = [
        Document(
            id="doc1",
            content="Векторные базы данных используются для хранения и поиска эмбеддингов. ChromaDB - одна из популярных векторных баз данных для Python разработки.",
            metadata={"category": "database", "language": "python"}
        ),
        Document(
            id="doc2",
            content="RAG (Retrieval-Augmented Generation) pipeline объединяет поиск релевантной информации с генерацией ответов. Это мощный подход для создания AI-ассистентов.",
            metadata={"category": "ai", "topic": "rag"}
        ),
        Document(
            id="doc3",
            content="SentenceTransformers предоставляет предобученные модели для создания качественных эмбеддингов текста. Модель all-MiniLM-L6-v2 подходит для большинства задач.",
            metadata={"category": "ml", "library": "sentence_transformers"}
        )
    ]

    # Добавление документов
    print("Добавление документов...")
    rag.add_documents_batch(test_documents)

    # Статистика
    print("\nСтатистика коллекции:")
    stats = rag.get_collection_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Поиск
    print("\nПоиск документов:")
    query = "как использовать векторные базы данных в Python"
    results = rag.search(query, n_results=3)

    for i, result in enumerate(results):
        print(f"\nРезультат {i+1}:")
        print(f"ID: {result.document_id}")
        print(f"Схожесть: {result.similarity_score:.3f}")
        print(f"Контент: {result.content[:100]}...")
        print(f"Метаданные: {result.metadata}")

    # Семантический поиск с переранжированием
    print("\nПоиск с переранжированием:")
    reranked = rag.semantic_search_with_reranking(query, n_results=5, rerank_top_k=2)

    for i, result in enumerate(reranked):
        print(f"\nПереранжированный результат {i+1}:")
        print(f"Схожесть: {result.similarity_score:.3f}")
        print(f"Контент: {result.content[:100]}...")
