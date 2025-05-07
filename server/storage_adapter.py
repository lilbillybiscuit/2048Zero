import os
import hashlib
import io
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import boto3
from botocore.client import Config
# Optional R2 dependency
logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """Abstract storage backend interface"""
    
    @abstractmethod
    def save_weights(self, model, revision: int) -> Tuple[str, str]:
        """Save model weights and return (path_or_key, sha256)"""
        pass
        
    @abstractmethod
    def get_weights_url(self, path_or_key: str, revision: int) -> str:
        """Get URL for accessing weights"""
        pass
        
    @abstractmethod
    def cleanup_old_weights(self, keep_revisions: int = 5) -> None:
        """Clean up old weight files"""
        pass


class LocalStorageBackend(StorageBackend):
    """Store weights in local filesystem"""
    
    def __init__(self, weights_dir: str = "weights", http_host: str = None, http_port: int = None):
        self.weights_dir = weights_dir
        self.http_host = http_host
        self.http_port = http_port
        os.makedirs(weights_dir, exist_ok=True)
        
    def save_weights(self, model, revision: int) -> Tuple[str, str]:
        """Save model weights to local file"""
        weights_path = os.path.join(self.weights_dir, f"r{revision}.pt")
        torch.save(model.state_dict(), weights_path)
        
        # Compute SHA256 hash
        sha256 = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        weights_sha256 = sha256.hexdigest()
        
        return weights_path, weights_sha256
        
    def get_weights_url(self, path_or_key: str, revision: int) -> str:
        assert self.http_host or self.http_port, "HTTP host and port are not set"
        host = "localhost" if self.http_host == "0.0.0.0" else self.http_host
        return f"http://{host}:{self.http_port}/weights/{revision}"
            
    def cleanup_old_weights(self, keep_revisions: int = 5) -> None:
        """Remove old weight files, keeping the most recent ones"""
        if not os.path.exists(self.weights_dir):
            return
            
        # List weight files and sort by revision (newest first)
        weight_files = []
        for f in os.listdir(self.weights_dir):
            if f.startswith("r") and f.endswith(".pt"):
                try:
                    rev = int(f[1:-3])  # Extract revision number from r{N}.pt
                    weight_files.append((rev, os.path.join(self.weights_dir, f)))
                except ValueError:
                    continue
                    
        weight_files.sort(reverse=True)
        
        # Keep the latest 'keep_revisions' files
        for rev, file_path in weight_files[keep_revisions:]:
            logger.info(f"Cleaning up old weights: {file_path}")
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove old weights file {file_path}: {e}")


class R2StorageBackend(StorageBackend):
    """Store weights in Cloudflare R2"""
    
    def __init__(
        self, 
        bucket_name: str,
        r2_account_id: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
        r2_public_url: Optional[str] = None,
        weights_dir: str = "weights"  # Local cache directory
    ):
        self.bucket_name = bucket_name
        self.r2_public_url = r2_public_url
        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)

        self.s3 = boto3.client(
            's3',
            endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name="auto"
        )
        
    def save_weights(self, model, revision: int) -> Tuple[str, str]:
        """Save model weights to R2 and local cache"""
        # First save locally
        weights_path = os.path.join(self.weights_dir, f"r{revision}.pt")
        torch.save(model.state_dict(), weights_path)
        
        # Compute SHA256 hash
        sha256 = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        weights_sha256 = sha256.hexdigest()
        
        # Upload to R2
        key = f"weights/r{revision}.pt"
        try:
            with open(weights_path, "rb") as f:
                self.s3.upload_fileobj(
                    f, 
                    self.bucket_name, 
                    key,
                    ExtraArgs={'ContentType': 'application/octet-stream'}
                )
            logger.info(f"Uploaded weights to R2: {key}")
        except Exception as e:
            raise Exception(f"Failed to upload weights to R2: {e}") from e

        return key, weights_sha256
        
    def get_weights_url(self, path_or_key: str, revision: int) -> str:
        assert self.r2_public_url, "R2 public URL is not set"
        return f"{self.r2_public_url.rstrip('/')}/weights/r{revision}.pt"

    def cleanup_old_weights(self, keep_revisions: int = 5) -> None:
        """Remove old weight files from R2 and local cache"""
        # List objects in R2
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix="weights/r"
        )

        if 'Contents' not in response:
            return

        # Extract revisions and sort (newest first)
        objects = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.startswith("weights/r") and key.endswith(".pt"):
                try:
                    rev = int(key[9:-3])  # Extract revision from weights/r{N}.pt
                    objects.append((rev, key))
                except ValueError:
                    continue

        objects.sort(reverse=True)

        # Delete old objects
        for rev, key in objects[keep_revisions:]:
            logger.info(f"Cleaning up old weights from R2: {key}")
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=key)
            except Exception as e:
                logger.warning(f"Failed to delete object {key} from R2: {e}")

        # Also cleanup local cache
        LocalStorageBackend(self.weights_dir).cleanup_old_weights(keep_revisions)

class StorageAdapter:
    """Storage adapter that selects the appropriate backend"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize backend based on configuration
        if config.get("use_r2", False):
            # Use R2 backend if configured and boto3 is available
            logger.info("Using R2 storage backend")
            self.backend = R2StorageBackend(
                bucket_name=config.get("r2_bucket", "2048Zero"),
                r2_account_id=config.get("r2_account_id", ""),
                r2_access_key_id=config.get("r2_access_key_id", ""),
                r2_secret_access_key=config.get("r2_secret_access_key", ""),
                r2_public_url=config.get("r2_public_url", None),
                weights_dir=config.get("weights_dir", "weights")
            )
        else:
            logger.info("Using local storage backend")
            self.backend = LocalStorageBackend(
                weights_dir=config.get("weights_dir", "weights"),
                http_host=config.get("host", None) if config.get("localhost_weights", False) else None,
                http_port=config.get("port", None) if config.get("localhost_weights", False) else None
            )
    
    def save_model(self, model, revision: int) -> Tuple[str, str, str]:
        """Save model weights and return (path_or_key, url, sha256)"""
        path_or_key, sha256 = self.backend.save_weights(model, revision)
        url = self.backend.get_weights_url(path_or_key, revision)
        return path_or_key, url, sha256
        
    def cleanup_old_models(self, keep_revisions: int = 5) -> None:
        """Clean up old model weights"""
        self.backend.cleanup_old_weights(keep_revisions)