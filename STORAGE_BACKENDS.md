# 2048-Zero Storage Backends Guide

This guide explains how to use different storage backends for model weights in the 2048-Zero training system.

## Storage Adapter Overview

The storage adapter provides a unified interface for storing and retrieving model weights across different backends:

1. **Local Filesystem**: Stores weights on the local filesystem and serves them via file:// URLs or HTTP
2. **Cloudflare R2**: Stores weights in Cloudflare's S3-compatible object storage service

The adapter ensures a consistent interface regardless of which backend is used, making it easy to switch between them with minimal code changes.

## Configuration

### Local Filesystem (Default)

The local filesystem backend is the default and simplest option. It stores weights in the local weights directory.

```yaml
storage:
  weights_dir: "weights"
  use_r2: false
  # Other settings...
```

You can also serve weights via HTTP for remote workers:

```yaml
server:
  localhost_weights: true  # Enable HTTP serving
  host: "your-server-ip"  # The host that workers will connect to
  port: 8000  # Port for the HTTP server
```

### Cloudflare R2

For production environments, Cloudflare R2 provides a scalable, globally-distributed storage solution.

To use R2:

1. Create an R2 bucket in your Cloudflare account
2. Create an API token with appropriate permissions
3. Configure the server to use R2:

```yaml
storage:
  use_r2: true
  r2_bucket: "your-bucket-name"
  r2_access_key: "your-access-key"
  r2_secret_key: "your-secret-key"
  r2_endpoint_url: "https://<account-id>.r2.cloudflarestorage.com"
  weights_dir: "weights"  # Local cache directory
```

#### Optional: Public Access

If you want to expose your weights via a public URL (e.g., through Cloudflare Workers or R2 public buckets):

```yaml
storage:
  # R2 configuration...
  r2_public_url: "https://weights.yourdomain.com"
```

## Command Line Options

You can override configuration settings via command line arguments:

```bash
python run_server.py --use-r2 --r2-bucket=your-bucket --r2-access-key=your-key --r2-secret-key=your-secret --r2-endpoint=your-endpoint
```

## Worker Integration

Workers automatically handle both storage backends. They will:

1. Download weights from the provided URL (file://, http://, or https://)
2. Verify the SHA256 hash for integrity
3. Load the model for self-play

## Fallback Mechanism

The storage adapter includes fallback mechanisms:

1. If R2 is configured but not available (missing boto3 or connectivity issues), the system falls back to local storage
2. Local copies of weights are always maintained as a cache, even when using R2
3. If generating presigned URLs fails, the adapter falls back to local file:// URLs

## Implementation Details

The storage adapter follows a clean design pattern:

1. **StorageBackend (ABC)**: Abstract base class defining the interface
2. **LocalStorageBackend**: Implements storage on the local filesystem
3. **R2StorageBackend**: Implements storage on Cloudflare R2
4. **StorageAdapter**: Main class that instantiates the appropriate backend

## Adding New Backends

You can extend the system by creating new backend classes that implement the StorageBackend interface:

```python
class NewStorageBackend(StorageBackend):
    def __init__(self, config):
        # Initialize with your settings
        pass
        
    def save_weights(self, weights_data, revision):
        # Implement weight saving
        pass
        
    def get_weights_url(self, weights_path, revision):
        # Implement URL generation
        pass
        
    def update_latest_info(self, revision, weights_url, weights_sha256, deadline):
        # Implement latest info updating
        pass
```

Then update the StorageAdapter to use your new backend based on configuration.

## Best Practices

1. **Local Development**: Use local filesystem backend for development
2. **Production**: Use R2 for production environments
3. **Security**: Keep R2 credentials secure and never commit them to version control
4. **Monitoring**: Add logging to track storage operations and detect issues
5. **Backups**: Regularly backup your weight files, especially for important model versions