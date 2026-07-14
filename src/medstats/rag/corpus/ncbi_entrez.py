"""Configure Biopython Entrez for NCBI E-utilities (SSL + API key)."""
from __future__ import annotations

import ssl
from typing import Optional
from urllib.request import urlopen as _stdlib_urlopen

from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
_configured = False


def ncbi_ssl_verify_enabled() -> bool:
    """True unless explicitly disabled or running in local dev/test."""
    if settings.NCBI_SSL_VERIFY is not None:
        return settings.NCBI_SSL_VERIFY
    return settings.ENV not in ("development", "test")


def _ssl_context() -> ssl.SSLContext:
    if not ncbi_ssl_verify_enabled():
        return ssl._create_unverified_context()
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def configure_entrez(email: Optional[str] = None) -> None:
    """Patch Bio.Entrez to use a working SSL context on macOS / proxied networks."""
    global _configured
    if _configured:
        return

    from Bio import Entrez

    Entrez.email = email or settings.NCBI_EMAIL
    if settings.NCBI_API_KEY:
        Entrez.api_key = settings.NCBI_API_KEY

    ctx = _ssl_context()

    def urlopen_with_ssl(request, *args, **kwargs):
        kwargs.setdefault("context", ctx)
        return _stdlib_urlopen(request, *args, **kwargs)

    Entrez.urlopen = urlopen_with_ssl  # type: ignore[method-assign]
    _configured = True

    if not ncbi_ssl_verify_enabled():
        logger.warning(
            "NCBI Entrez SSL verification disabled for local dev",
            env=settings.ENV,
        )
