"""File storage — S3 in production, local filesystem in development."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse

from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class StorageService:
    """Upload and resolve file URLs via S3 or local disk."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    @property
    def use_s3(self) -> bool:
        return bool(
            self.settings.S3_BUCKET_NAME
            and self.settings.S3_ACCESS_KEY_ID
            and self.settings.S3_SECRET_ACCESS_KEY
        )

    @property
    def client(self):
        if not self.use_s3:
            return None
        if self._client is None:
            import boto3

            kwargs = {
                "aws_access_key_id": self.settings.S3_ACCESS_KEY_ID,
                "aws_secret_access_key": self.settings.S3_SECRET_ACCESS_KEY,
                "region_name": self.settings.S3_REGION or "us-east-1",
            }
            if self.settings.S3_ENDPOINT_URL:
                kwargs["endpoint_url"] = self.settings.S3_ENDPOINT_URL
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def upload_file(self, local_path: str, key: str, content_type: Optional[str] = None) -> str:
        """Upload a file and return its public or API-relative URL."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(local_path)

        if not self.use_s3:
            return f"/api/files/{key}"

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.upload_file(
            local_path,
            self.settings.S3_BUCKET_NAME,
            key,
            ExtraArgs=extra_args or None,
        )
        return self._public_url(key)

    def get_presigned_url(self, key: str, expires: int = 3600) -> str:
        if not self.use_s3:
            return f"/api/files/{key}"
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expires,
        )

    def file_exists(self, key: str) -> bool:
        if not self.use_s3:
            local = os.path.join(self.settings.UPLOAD_DIR, key)
            return os.path.exists(local)
        try:
            self.client.head_object(Bucket=self.settings.S3_BUCKET_NAME, Key=key)
            return True
        except Exception:
            return False

    def _public_url(self, key: str) -> str:
        endpoint = (self.settings.S3_ENDPOINT_URL or "").rstrip("/")
        bucket = self.settings.S3_BUCKET_NAME
        region = self.settings.S3_REGION or "us-east-1"

        if endpoint and "amazonaws.com" not in endpoint:
            return f"{endpoint}/{bucket}/{key}"

        if region == "us-east-1":
            return f"https://{bucket}.s3.amazonaws.com/{key}"
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

    def resolve_local_path(self, key_or_url: str) -> Optional[str]:
        """Resolve a storage key or URL to a local filesystem path."""
        if not key_or_url:
            return None
        if key_or_url.startswith("/api/files/"):
            rel = key_or_url.removeprefix("/api/files/")
            return os.path.join(self.settings.UPLOAD_DIR, rel)
        if key_or_url.startswith("http"):
            parsed = urlparse(key_or_url)
            path = parsed.path.lstrip("/")
            if path.startswith(f"{self.settings.S3_BUCKET_NAME}/"):
                path = path.split("/", 1)[1]
            return os.path.join(self.settings.UPLOAD_DIR, path)
        return os.path.join(self.settings.UPLOAD_DIR, key_or_url)


@lru_cache()
def get_storage_service() -> StorageService:
    return StorageService()
