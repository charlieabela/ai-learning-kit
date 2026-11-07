"""HTTPS uses the operating system's certificates, so the Python uv installs on macOS can verify servers."""
import ssl


def test_the_context_verifies_servers():
    from ai_learning_kit.core import tls
    tls.context.cache_clear()
    ctx = tls.context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_ssl_cert_file_still_wins(monkeypatch, tmp_path):
    from ai_learning_kit.core import tls
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "none.pem"))
    tls.context.cache_clear()
    try:
        assert type(tls.context()) is ssl.SSLContext          # Python's own, which reads SSL_CERT_FILE
    finally:
        monkeypatch.delenv("SSL_CERT_FILE")
        tls.context.cache_clear()


def test_without_truststore_certifi_is_used(monkeypatch):
    import builtins
    from ai_learning_kit.core import tls
    real = builtins.__import__

    def no_truststore(name, *a, **k):
        if name == "truststore":
            raise ImportError("no truststore")
        return real(name, *a, **k)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_truststore)
    tls.context.cache_clear()
    try:
        ctx = tls.context()
        assert ctx.cert_store_stats()["x509_ca"] > 50                  # Mozilla's list was loaded
    finally:
        tls.context.cache_clear()
