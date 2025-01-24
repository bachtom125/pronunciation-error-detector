import asyncio
from cachetools import TTLCache

class CacheManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CacheManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Initialize the cache and lock only once
        self.cache = TTLCache(maxsize=100, ttl=300)
        self.lock = asyncio.Lock()

    async def set(self, key, value):
        async with self.lock:
            self.cache[key] = value

    async def get(self, key):
        async with self.lock:
            return self.cache.get(key, None)
    
    async def contains(self, key):
        async with self.lock:
            return key in self.cache
        
    async def delete(self, key):
        async with self.lock:
            if key in self.cache:
                del self.cache[key]

    def set_without_lock(self, key, value):
        self.cache[key] = value

    def get_without_lock(self, key):
        return self.cache.get(key, None)
    
    def contains_without_lock(self, key):
        return key in self.cache
    
    def delete_without_lock(self, key):
        if key in self.cache:
            del self.cache[key]

audio_cache = CacheManager()
