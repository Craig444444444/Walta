"""
Vector store implementation for Walta Framework.
Last Updated: 2025-07-05 03:23:13 UTC
Author: Craig444444444
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass

@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    collection_name: str = "walta_vectors"
    dimension: int = 1536  # Default for OpenAI ada-002 embeddings
    similarity_metric: str = "cosine"
    persist_directory: str = "data/vector_store"
    max_batch_size: int = 100
    metadata: Dict[str, Any] = None

class WaltaVectorStore:
    """
    Vector store for managing and searching vector embeddings.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize vector store with configuration."""
        self.config = config
        self._initialize_store()
        logger.info(
            f"WaltaVectorStore initialized with collection '{config.collection_name}'"
        )

    def _initialize_store(self):
        """Initialize the vector store backend."""
        try:
            # Ensure persist directory exists
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize Chroma client
            self.client = chromadb.Client(Settings(
                persist_directory=self.config.persist_directory,
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "dimension": self.config.dimension,
                    "similarity": self.config.similarity_metric,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    **(self.config.metadata or {})
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreError(f"Vector store initialization failed: {e}")

    async def add_data(
        self,
        document_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add data to the vector store.
        """
        try:
            if len(embedding) != self.config.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.config.dimension}, "
                    f"got {len(embedding)}"
                )
            
            # Add timestamp to metadata
            metadata = metadata or {}
            metadata.update({
                "added_at": datetime.utcnow().isoformat(),
                "content_length": len(content)
            })
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            logger.debug(f"Added document {document_id} to vector store")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to add data to vector store: {e}")
            raise VectorStoreError(f"Failed to add data: {e}")

    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the store.
        """
        try:
            if len(query_embedding) != self.config.dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch. Expected {self.config.dimension}, "
                    f"got {len(query_embedding)}"
                )
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Process results
            processed_results = []
            for i, (doc_id, score, metadata, document) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0],
                results['documents'][0]
            )):
                if score >= score_threshold:
                    processed_results.append({
                        "id": doc_id,
                        "score": 1.0 - score,  # Convert distance to similarity score
                        "metadata": metadata,
                        "content": document
                    })
            
            logger.debug(
                f"Found {len(processed_results)} similar documents above threshold"
            )
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            raise VectorStoreError(f"Search failed: {e}")

    async def update_data(
        self,
        document_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Update existing data in the vector store.
        """
        try:
            # Add update timestamp to metadata
            metadata = metadata or {}
            metadata.update({
                "updated_at": datetime.utcnow().isoformat(),
                "content_length": len(content)
            })
            
            # Update in collection
            self.collection.update(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            logger.debug(f"Updated document {document_id} in vector store")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to update data in vector store: {e}")
            raise VectorStoreError(f"Update failed: {e}")

    async def delete_data(self, document_id: str) -> str:
        """
        Delete data from the vector store.
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted document {document_id} from vector store")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to delete data from vector store: {e}")
            raise VectorStoreError(f"Deletion failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "dimension": self.config.dimension,
                "similarity_metric": self.config.similarity_metric,
                "collection_name": self.config.collection_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            raise VectorStoreError(f"Failed to get stats: {e}")

    async def backup(self, backup_dir: str) -> str:
        """
        Create a backup of the vector store.
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                backup_dir,
                f"walta_vectors_backup_{timestamp}"
            )
            
            # Ensure backup directory exists
            os.makedirs(backup_dir, exist_ok=True)
            
            # Persist current state
            self.client.persist()
            
            # Copy persistence directory to backup location
            import shutil
            shutil.copytree(
                self.config.persist_directory,
                backup_path
            )
            
            logger.info(f"Created vector store backup at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create vector store backup: {e}")
            raise VectorStoreError(f"Backup failed: {e}")

    async def restore(self, backup_path: str) -> bool:
        """
        Restore vector store from backup.
        """
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup not found at {backup_path}")
            
            # Close current client
            self.client.persist()
            
            # Copy backup to persistence directory
            import shutil
            shutil.rmtree(self.config.persist_directory)
            shutil.copytree(backup_path, self.config.persist_directory)
            
            # Reinitialize store
            self._initialize_store()
            
            logger.info(f"Restored vector store from backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore vector store from backup: {e}")
            raise VectorStoreError(f"Restore failed: {e}")
