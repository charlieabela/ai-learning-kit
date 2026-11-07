"""Which certificates HTTPS connections trust, the same on every computer.

Python's default settings use the certificates OpenSSL was built to look for. On some computers there are none: the
Python that uv installs on macOS, for example, fails every HTTPS connection with CERTIFICATE_VERIFY_FAILED. So the
kit trusts what the operating system trusts (truststore: the macOS keychain, the Windows certificate store, or the
Linux system store), which also covers a university network that inspects traffic with its own certificate. Without
truststore it uses Mozilla's list (certifi). SSL_CERT_FILE or SSL_CERT_DIR, when set, still win.
"""
from __future__ import annotations

import os
import ssl
from functools import lru_cache


@lru_cache(maxsize=1)
def context() -> ssl.SSLContext:
    """The SSL context for the kit's HTTPS connections (language models, dataset downloads)."""
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context()
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:                       # not installed, or the system store cannot be read
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def opener(*handlers):
    """urllib's opener with the kit's certificates; it honours HTTP(S)_PROXY like any client."""
    import urllib.request
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context()), *handlers)
