"""RG Mining Service configuration."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service
    SERVICE_NAME: str = "rg-mining"
    SERVICE_VERSION: str = "0.1.0"

    # Database (for persisting miner stats, task history)
    DATABASE_URL: str = os.getenv(
        "MINING_DATABASE_URL",
        f"postgresql+asyncpg://{os.getenv('MINING_DB_USER', 'postgres')}:"
        f"{os.getenv('MINING_DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('MINING_DB_HOST', 'db')}:"
        f"{os.getenv('MINING_DB_PORT', '5432')}/"
        f"{os.getenv('MINING_DB_NAME', 'resonant_mining')}?ssl=require"
    )

    # External blockchain API (RG_external_blockchain service)
    EXTERNAL_BLOCKCHAIN_URL: str = os.getenv(
        "EXTERNAL_BLOCKCHAIN_URL", "http://external_blockchain_service:8000"
    )

    # P2P network
    P2P_PORT: int = int(os.getenv("P2P_PORT", "8600"))
    P2P_BOOTSTRAP_NODES: str = os.getenv("P2P_BOOTSTRAP_NODES", "[]")

    # Parameter server
    STALENESS_ALPHA: float = float(os.getenv("STALENESS_ALPHA", "0.5"))
    MAX_STALENESS: int = int(os.getenv("MAX_STALENESS", "50"))
    MIN_MINERS_PER_ROUND: int = int(os.getenv("MIN_MINERS_PER_ROUND", "1"))

    # Genesis model defaults
    DEFAULT_MODEL_ID: str = "resonant-seed-1b"
    DEFAULT_NUM_DATA_SHARDS: int = 100
    DEFAULT_NUM_WEIGHT_SHARDS: int = 10

    # IPFS / storage
    IPFS_BASE_URL: str = os.getenv("IPFS_BASE_URL", "ipfs://")

    # Crypto service (for rewards)
    CRYPTO_SERVICE_URL: str = os.getenv(
        "CRYPTO_SERVICE_URL", "http://crypto_service:8000"
    )

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/5")

    class Config:
        env_file = ".env"


settings = Settings()
