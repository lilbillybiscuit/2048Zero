#!/usr/bin/env python
"""
R2 checkpoint utility for 2048-Zero

This script provides utilities to download, upload, and manage checkpoints in R2 storage.
"""

import os
import sys
import json
import argparse
import requests
from typing import Optional, Tuple, Dict, Any

def download_checkpoint(url: str, output_dir: str = "checkpoints") -> Optional[str]:
    """
    Download a checkpoint file from a URL
    
    Args:
        url: URL to download from
        output_dir: Directory to save to
        
    Returns:
        Path to downloaded file or None if failed
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract filename from URL
        local_filename = os.path.join(output_dir, url.split("/")[-1])
        
        print(f"Downloading {url} to {local_filename}...")
        
        # Download with progress tracking
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with open(local_filename, 'wb') as f:
            if total_size == 0:
                print("Warning: Unknown file size")
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Update progress
                        progress = downloaded / total_size * 100
                        sys.stdout.write(f"\rProgress: {progress:.1f}% ({downloaded/1024/1024:.1f} MB)")
                        sys.stdout.flush()
                print()  # Newline after progress
        
        print(f"Successfully downloaded checkpoint to {local_filename}")
        return local_filename
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        return None

def get_signed_url(bucket: str, key: str, access_key: str, secret_key: str, 
                   endpoint_url: str, region: str = "auto", expiration: int = 3600) -> Optional[str]:
    """
    Generate a signed URL for accessing a private R2/S3 object
    
    Args:
        bucket: Bucket name
        key: Object key (path in bucket)
        access_key: R2/S3 access key ID
        secret_key: R2/S3 secret access key
        endpoint_url: R2/S3 endpoint URL
        region: Region name (default: "auto" for R2)
        expiration: URL expiration time in seconds (default: 1 hour)
        
    Returns:
        Signed URL or None if failed
    """
    try:
        import boto3
        from botocore.client import Config
        
        # Create S3 client for R2
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version='s3v4')
        )
        
        # Generate a signed URL
        signed_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        
        return signed_url
    except Exception as e:
        print(f"Error generating signed URL: {e}")
        return None

def handle_r2_url(r2_url: str, env_vars: Dict[str, str] = None) -> Optional[str]:
    """
    Handle an R2 URL and return an HTTP URL for downloading
    
    Args:
        r2_url: R2 URL in format r2://bucket/key
        env_vars: Environment variables or dict containing credentials
        
    Returns:
        HTTP URL or None if failed
    """
    if not r2_url.startswith("r2://"):
        return r2_url
    
    try:
        # Parse bucket and key
        _, bucket_key = r2_url.split("://", 1)
        bucket, key = bucket_key.split("/", 1)
        
        # Get credentials
        env = env_vars or os.environ
        access_key = env.get("R2_ACCESS_KEY")
        secret_key = env.get("R2_SECRET_KEY")
        endpoint_url = env.get("R2_ENDPOINT")
        
        if not all([access_key, secret_key, endpoint_url]):
            print("Error: R2 credentials not found in environment variables")
            print("Please set R2_ACCESS_KEY, R2_SECRET_KEY, and R2_ENDPOINT")
            return None
        
        # Generate signed URL
        return get_signed_url(bucket, key, access_key, secret_key, endpoint_url)
    except Exception as e:
        print(f"Failed to handle R2 URL: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="R2 checkpoint downloader for 2048-Zero")
    
    parser.add_argument("url", help="URL or R2 URL to download (e.g., r2://bucket/key)")
    parser.add_argument("--output-dir", "-o", default="checkpoints", 
                        help="Directory to save checkpoint to")
    
    args = parser.parse_args()
    
    # Handle R2 URL if needed
    url = args.url
    if url.startswith("r2://"):
        print(f"Handling R2 URL: {url}")
        url = handle_r2_url(url)
        if not url:
            sys.exit(1)
    
    # Download the file
    local_path = download_checkpoint(url, args.output_dir)
    if not local_path:
        sys.exit(1)
    
    # Also try downloading run data with same filename pattern
    try:
        run_url = url.replace(".pth", "_run.json")
        run_path = os.path.join(args.output_dir, os.path.basename(run_url))
        print(f"Attempting to download run data from: {run_url}")
        run_response = requests.get(run_url, timeout=10)
        
        if run_response.status_code == 200:
            with open(run_path, 'wb') as f:
                f.write(run_response.content)
            print(f"Successfully downloaded run data to {run_path}")
        else:
            print(f"No run data found (status code: {run_response.status_code})")
    except Exception as e:
        print(f"Note: Could not download run data: {e}")
    
    print(f"Checkpoint downloaded to: {local_path}")
    print(f"Use --resume-from={local_path} to resume training from this checkpoint")

if __name__ == "__main__":
    main()