"""Improved cache manager with better error handling and type safety."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic
from cachetools import TTLCache

from core.exceptions import CacheError
from config.settings import Settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheInterface(ABC, Generic[T]):
    """Abstract interface for cache operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Retrieve value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T) -> None:
        """Store value in cache."""
        pass
    
    @abstractmethod
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove key from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass


class TTLCacheManager(CacheInterface[T]):
    """Thread-safe TTL cache manager with improved error handling."""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, settings: Settings = None):
        if hasattr(self, '_initialized'):
            return
            
        self._settings = settings or Settings()
        self._cache = TTLCache(
            maxsize=self._settings.model.cache_maxsize,
            ttl=self._settings.model.cache_ttl
        )
        self._lock = asyncio.Lock()
        self._initialized = True
        logger.info(f"Cache initialized with maxsize={self._settings.model.cache_maxsize}, ttl={self._settings.model.cache_ttl}")
    
    async def get(self, key: str) -> Optional[T]:
        """Retrieve value from cache with error handling."""
        try:
            async with self._lock:
                value = self._cache.get(key)
                if value is not None:
                    logger.debug(f"Cache hit for key: {key}")
                else:
                    logger.debug(f"Cache miss for key: {key}")
                return value
        except Exception as e:
            logger.error(f"Error retrieving from cache for key {key}: {e}")
            raise CacheError(f"Failed to retrieve from cache: {e}")
    
    async def set(self, key: str, value: T) -> None:
        """Store value in cache with error handling."""
        try:
            async with self._lock:
                self._cache[key] = value
                logger.debug(f"Cached value for key: {key}")
        except Exception as e:
            logger.error(f"Error storing to cache for key {key}: {e}")
            raise CacheError(f"Failed to store in cache: {e}")
    
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            async with self._lock:
                exists = key in self._cache
                logger.debug(f"Cache contains check for key {key}: {exists}")
                return exists
        except Exception as e:
            logger.error(f"Error checking cache for key {key}: {e}")
            raise CacheError(f"Failed to check cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Remove key from cache."""
        try:
            async with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"Deleted key from cache: {key}")
        except Exception as e:
            logger.error(f"Error deleting from cache for key {key}: {e}")
            raise CacheError(f"Failed to delete from cache: {e}")
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            async with self._lock:
                self._cache.clear()
                logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise CacheError(f"Failed to clear cache: {e}")
    
    # Synchronous methods for backwards compatibility
    def get_sync(self, key: str) -> Optional[T]:
        """Synchronous get (for backwards compatibility)."""
        return self._cache.get(key)
    
    def set_sync(self, key: str, value: T) -> None:
        """Synchronous set (for backwards compatibility)."""
        self._cache[key] = value
    
    def contains_sync(self, key: str) -> bool:
        """Synchronous contains check (for backwards compatibility)."""
        return key in self._cache
    
    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
            "ttl": self._cache.ttl,
            "hits": getattr(self._cache, 'hits', 0),
            "misses": getattr(self._cache, 'misses', 0)
        }


# Global cache instance
audio_cache = TTLCacheManager() 