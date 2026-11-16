"""Verified HTTPS using optional native trust stores or an explicit CA bundle.

Copied from emotion_recognition/dependencies/torch-cu118-cp311-linux_x86_64/tls.py (the model and dataset
Release downloaders use the same code); only the hints differ, because this folder's scripts take no
command-line options. It deliberately does not patch ssl globally or offer an insecure mode.
"""
import importlib
import os
from pathlib import Path
import ssl
import sys


def create_ssl_context(ca_bundle=None):
    """Use native truststore when installed; retain normal TLS and hostname checks.

    An explicit CA bundle (CA_BUNDLE in download.py) takes precedence over SSL_CERT_FILE. Its certificates
    supplement the selected context's default trust roots.
    """
    try:
        truststore = importlib.import_module("truststore")
    except ImportError:
        context = ssl.create_default_context()
    else:
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # Native truststore does not necessarily read OpenSSL's environment paths.
        # Load any configured CA explicitly below for consistent behavior.
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    selected = ca_bundle if ca_bundle is not None else os.environ.get("SSL_CERT_FILE")
    if selected:
        path = Path(selected).expanduser()
        if not path.is_file():
            raise ValueError("CA bundle does not exist or is not a file: " + str(path))
        try:
            context.load_verify_locations(cafile=str(path))
        except (OSError, ssl.SSLError) as exc:
            raise ValueError("Unable to load CA bundle {}: {}".format(path, exc)) from exc
    return context


def certificate_error(exc):
    """Recognize SSL verification failures even when wrapped by urllib."""
    pending = [exc]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        if isinstance(current, ssl.SSLError) and getattr(current, "reason", None) == "CERTIFICATE_VERIFY_FAILED":
            return True
        for attribute in ("reason", "__cause__", "__context__"):
            nested = getattr(current, attribute, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def certificate_error_message(exc):
    """Explain how to trust company roots without weakening TLS verification."""
    version_hint = ("Install native certificate-store support in this Python environment with "
                    "`python -m pip install truststore` (Python 3.10+), then rerun download.py. "
                    if sys.version_info >= (3, 10) else
                    "Native certificate-store support via truststore requires Python 3.10+; "
                    "use a newer environment or provide the company CA bundle below. ")
    return ("HTTPS certificate verification failed: {}. "
            "On a company-managed machine, the company root CA may be trusted by the operating system "
            "but absent from this Python environment. {}"
            "Alternatively, obtain the company root/intermediate CA PEM bundle from IT and set "
            "CA_BUNDLE = \"/path/to/company-ca.pem\" at the top of download.py, or set SSL_CERT_FILE to that file. "
            "If package installation itself needs that CA, use "
            "`python -m pip install --cert /path/to/company-ca.pem truststore`. "
            "Certificate and hostname verification remain enabled. Existing verified files and "
            "resumable .download parts are preserved.").format(exc, version_hint)
