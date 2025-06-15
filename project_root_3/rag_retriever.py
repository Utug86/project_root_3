import faiss
import numpy as np
import json
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Any, Dict, List, Optional, Union
import hashlib
import logging
import datetime

from rag_file_utils import extract_text_from_file

def notify_admin(message: str) -> None:
    logging.warning(f"[ADMIN NOTIFY] {message}")

class HybridRetriever:
    """
    Гибридный ретривер: поиск релевантных чанков по FAISS-индексу и кросс-энкодеру.
    Гарантирует валидацию индекса, atomic запись, резервное копирование, обработку edge-cases.
    """
    INDEX_VERSION = "1.2"

    def __init__(
        self,
        emb_model: str,
        cross_model: str,
        index_file: Path,
        context_file: Path,
        inform_dir: Path,
        chunk_size: int,
        overlap: int,
        top_k_title: int,
        top_k_faiss: int,
        top_k_final: int,
        usage_tracker,
        logger: logging.Logger
    ):
        self.emb_model = emb_model
        self.cross_model = cross_model
        self.index_file = Path(index_file)
        self.context_file = Path(context_file)
        self.inform_dir = Path(inform_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k_title = top_k_title
        self.top_k_faiss = top_k_faiss
        self.top_k_final = top_k_final
        self.usage_tracker = usage_tracker
        self.logger = logger

        self.sentencemodel = SentenceTransformer(self.emb_model)
        self.crossencoder = CrossEncoder(self.cross_model)
        self.faiss_index: Optional[faiss.Index] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None
        self.index_metadata: Dict[str, Any] = {}

        self._try_load_or_build_indices()

    def _get_index_signature(self) -> str:
        conf = {
            "version": self.INDEX_VERSION,
            "emb_model": self.emb_model,
            "cross_model": self.cross_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "inform_dir_hash": self._dir_hash(self.inform_dir)
        }
        return hashlib.sha256(json.dumps(conf, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _dir_hash(directory: Path) -> str:
        if not directory.exists():
            return ""
        files = sorted([(str(f), f.stat().st_mtime) for f in directory.iterdir() if f.is_file()])
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    def _try_load_or_build_indices(self) -> None:
        self.logger.info("Initializing HybridRetriever...")
        rebuild_needed = False
        if self.index_file.exists() and self.context_file.exists():
            try:
                self._load_indices()
                idx_sig = self.index_metadata.get("index_signature")
                expected_sig = self._get_index_signature()
                if idx_sig != expected_sig:
                    self.logger.warning("Index signature mismatch. Rebuilding index...")
                    notify_admin("HybridRetriever: Index signature mismatch, rebuild triggered.")
                    rebuild_needed = True
            except Exception as e:
                self.logger.warning(f"Failed to load indices: {e}. Rebuilding...")
                notify_admin(f"HybridRetriever: Failed to load indices: {e}. Rebuilding.")
                rebuild_needed = True
        else:
            self.logger.info("No existing indices found. Building new ones...")
            rebuild_needed = True
        if rebuild_needed:
            self._build_indices()

    def _load_indices(self) -> None:
        self.logger.info("Loading indices...")
        try:
            with open(self.context_file, 'r', encoding='utf-8') as f:
                context_json = json.load(f)
            if not isinstance(context_json, dict) or "metadata" not in context_json:
                raise ValueError("context_file missing 'metadata' field")
            self.metadata = context_json["metadata"]
            self.index_metadata = context_json.get("index_metadata", {})
            if not isinstance(self.metadata, list) or len(self.metadata) == 0:
                raise ValueError("Metadata must be a non-empty list")
            # Ensure fields exist
            for item in self.metadata:
                item.setdefault('tokens', item['chunk'].split())
                item.setdefault('created_at', None)
                item.setdefault('source', None)
            self.faiss_index = faiss.read_index(str(self.index_file))
            # Validate dimension
            sample_chunk = self.metadata[0]['chunk']
            emb_sample = self.sentencemodel.encode([sample_chunk], convert_to_tensor=False, normalize_embeddings=True)
            if self.faiss_index.d != emb_sample.shape[1]:
                self.logger.critical(
                    f"Embedding dimension mismatch: FAISS index {self.faiss_index.d}, model {emb_sample.shape[1]}"
                )
                notify_admin("HybridRetriever: Embedding dimension mismatch, index rebuild required")
                raise RuntimeError("Embedding dimension mismatch — rebuild index!")
            self.logger.info(f"Loaded {len(self.metadata)} chunks")
            self.logger.info(f"Index metadata: {self.index_metadata}")
        except Exception as e:
            self.logger.critical(f"Error loading indices: {e}")
            notify_admin(f"HybridRetriever: Error loading indices: {e}")
            raise

    def _normalize_text(self, text: str) -> str:
        """
        Базовая нормализация текста (html → plain, lower, clean).
        """
        import re
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^а-яa-z0-9\s\.,:;!\?\(\)\[\]\'\"-]', '', text)
        return text.strip()

    def _semantic_deduplicate(self, chunks: List[Dict[str, Any]], threshold: float = 0.91) -> List[Dict[str, Any]]:
        """
        Семантическая дедупликация: удаление дубликатов по cosine similarity эмбеддингов.
        Для больших коллекций (N>10000) пропускается.
        """
        if len(chunks) < 2:
            return chunks
        if len(chunks) > 10000:
            self.logger.warning("Skipping semantic deduplication due to excessive collection size")
            return chunks
        texts = [c['chunk'] for c in chunks]
        embs = self.sentencemodel.encode(texts, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
        embs = np.asarray(embs, dtype='float32')
        to_keep = []
        already_used = set()
        for i in range(len(chunks)):
            if i in already_used:
                continue
            to_keep.append(chunks[i])
            for j in range(i + 1, len(chunks)):
                if j in already_used:
                    continue
                sim = np.dot(embs[i], embs[j])
                if sim > threshold:
                    already_used.add(j)
        self.logger.info(f"Semantic deduplication: {len(chunks)} → {len(to_keep)} unique chunks (threshold={threshold})")
        return to_keep

    def _build_indices(self) -> None:
        self.logger.info("Building new indices...")
        metadata = []
        inform_files = [f for f in self.inform_dir.iterdir()
                        if f.suffix.lower() in [".txt", ".html", ".csv", ".xlsx", ".xlsm", ".doc", ".docx", ".pdf"]]
        if not inform_files:
            notify_admin(f"HybridRetriever: No suitable files in {self.inform_dir}")
            raise RuntimeError(f"No suitable files in {self.inform_dir}")
        self.logger.info(f"Found {len(inform_files)} files to process")
        index_time = datetime.datetime.utcnow().isoformat()
        for file in inform_files:
            try:
                title = file.stem.lower()
                text = extract_text_from_file(file)
                if not text or not text.strip():
                    self.logger.warning(f"Empty or unreadable file: {file}")
                    continue
                text = self._normalize_text(text)
                words = text.split()
                chunks = []
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk = ' '.join(words[i:i + self.chunk_size])
                    if len(chunk.strip()) < 10:
                        continue
                    tokens = chunk.split()
                    if any(chunk == m['chunk'] for m in chunks):
                        continue
                    chunks.append({'title': title, 'chunk': chunk, 'tokens': tokens,
                                   'created_at': index_time, 'source': str(file)})
                if not chunks:
                    continue
                deduped_chunks = self._semantic_deduplicate(chunks, threshold=0.91)
                metadata.extend(deduped_chunks)
                self.logger.info(f"Processed {file.name}: {len(words)} words -> {len(deduped_chunks)} unique chunks")
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {e}")
                notify_admin(f"HybridRetriever: Error processing file {file}: {e}")
        if not metadata:
            notify_admin("HybridRetriever: No valid chunks created from files")
            raise RuntimeError("No valid chunks created from files")
        metadata = self._semantic_deduplicate(metadata, threshold=0.91)
        self.logger.info(f"Total unique chunks after global deduplication: {len(metadata)}")
        try:
            # Backup before overwrite
            if self.index_file.exists():
                shutil.copy2(self.index_file, self.index_file.with_suffix('.bak'))
            if self.context_file.exists():
                shutil.copy2(self.context_file, self.context_file.with_suffix('.bak'))
            texts = [f"{m['title']}: {m['chunk']}" for m in metadata]
            self.logger.info("Generating embeddings...")
            embs = self.sentencemodel.encode(texts, convert_to_tensor=False, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
            embs = np.asarray(embs, dtype='float32')
            dim = embs.shape[1]
            if len(metadata) > 10000:
                self.logger.info("Large dataset: using HNSW index.")
                faiss_index = faiss.IndexHNSWFlat(dim, 32)
            else:
                faiss_index = faiss.IndexFlatL2(dim)
            faiss_index.add(embs)
            # Atomic write
            tmp_index = self.index_file.with_suffix('.tmp')
            faiss.write_index(faiss_index, str(tmp_index))
            tmp_index.replace(self.index_file)
            self.faiss_index = faiss_index
            self.index_metadata = {
                "index_signature": self._get_index_signature(),
                "version": self.INDEX_VERSION,
                "emb_model": self.emb_model,
                "cross_model": self.cross_model,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "created_at": index_time,
                "num_chunks": len(metadata),
                "inform_dir": str(self.inform_dir),
            }
            context_json = {
                "metadata": metadata,
                "index_metadata": self.index_metadata
            }
            tmp_context = self.context_file.with_suffix('.tmp')
            with open(tmp_context, 'w', encoding='utf-8') as f:
                json.dump(context_json, f, ensure_ascii=False, indent=2)
            tmp_context.replace(self.context_file)
            self.metadata = metadata
            self.logger.info(f"Indices built and saved: {len(metadata)} unique chunks, index type: {type(self.faiss_index)}")
            self.logger.info(f"Index metadata: {self.index_metadata}")
            notify_admin(f"HybridRetriever: Index rebuilt. {len(metadata)} unique chunks.")
        except Exception as e:
            self.logger.error(f"Error building or saving indices: {e}")
            notify_admin(f"HybridRetriever: Error building index: {e}")
            raise

    def retrieve(self, query: str, return_chunks: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Возвращает релевантный контекст (или чанки) по запросу.
        :param query: Запрос пользователя
        :param return_chunks: Если True, возвращает список чанков, иначе — строку
        """
        self.logger.info(f"Retrieving context for query: '{query}'")
        if self.faiss_index is None or self.metadata is None or len(self.metadata) == 0:
            self.logger.error("Index not loaded or metadata is empty")
            notify_admin("HybridRetriever: Retrieval failed — index not loaded or metadata is empty")
            return "" if not return_chunks else []
        query_emb = self.sentencemodel.encode([query], convert_to_tensor=False, normalize_embeddings=True)
        query_emb = np.asarray(query_emb, dtype='float32')
        if self.faiss_index.d != query_emb.shape[1]:
            self.logger.critical(
                f"Embedding dimension mismatch: FAISS index {self.faiss_index.d}, query_emb {query_emb.shape[1]}"
            )
            notify_admin("HybridRetriever: Embedding dimension mismatch, index rebuild required")
            raise RuntimeError("Embedding dimension mismatch — rebuild index!")
        D, I = self.faiss_index.search(query_emb, self.top_k_faiss)
        I = I[0]
        D = D[0]
        if not len(I):
            self.logger.warning("No relevant chunks found in index")
            return "" if not return_chunks else []
        candidates = []
        for idx, dist in zip(I, D):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta['faiss_dist'] = float(dist)
            candidates.append(meta)
        for c in candidates:
            c['usage_penalty'] = self.usage_tracker.get_usage_penalty(
                c['chunk'], c['title']) if self.usage_tracker else 0.0
        candidates.sort(key=lambda x: (x['faiss_dist'] + x['usage_penalty']))
        rerank_candidates = candidates[:self.top_k_final * 2]
        ce_scores = []
        try:
            if self.crossencoder:
                pairs = [[query, c['chunk']] for c in rerank_candidates]
                ce_scores = self.crossencoder.predict(pairs)
                for c, score in zip(rerank_candidates, ce_scores):
                    c['cross_score'] = float(score)
                rerank_candidates.sort(key=lambda x: -x.get('cross_score', 0))
            else:
                for c in rerank_candidates:
                    c['cross_score'] = 0.0
        except Exception as e:
            self.logger.error(f"Cross-encoder rerank failed: {e}")
            for c in rerank_candidates:
                c['cross_score'] = 0.0
        selected = []
        titles = set()
        for c in rerank_candidates:
            if len(selected) >= self.top_k_final:
                break
            if c['title'] not in titles or self.top_k_final > len(rerank_candidates):
                selected.append(c)
                titles.add(c['title'])
        result = "\n\n".join([f"[{c['title']}] {c['chunk']}" for c in selected])
        self.logger.info(f"Retrieved {len(selected)} chunks from index for query '{query}'")
        return selected if return_chunks else result

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по индексу.
        """
        stats = {
            "num_chunks": len(self.metadata) if self.metadata else 0,
            "num_files": len(set(m['source'] for m in self.metadata)) if self.metadata else 0,
            "unique_titles": len(set(m['title'] for m in self.metadata)) if self.metadata else 0,
            "index_type": type(self.faiss_index).__name__ if self.faiss_index else None,
            "index_metadata": self.index_metadata,
        }
        self.logger.info(f"Index stats: {stats}")
        return stats

    def rebuild_index(self) -> None:
        """
        Принудительное перестроение индекса (с backup).
        """
        self.logger.info("Manual index rebuild triggered...")
        notify_admin("Manual HybridRetriever index rebuild triggered by user.")
        self._build_indices()
