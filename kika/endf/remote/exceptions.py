"""Custom exceptions for ENDF remote download functionality."""


class ENDFRemoteError(Exception):
    """Base class of every error the remote layer raises on its own.

    Catch it to handle the errors below at once. A failed copy or write to the
    destination the caller gave (``save_to``) raises a bare ``OSError``, which is
    not an ``ENDFRemoteError``.
    """

    pass


class IsotopeNotFoundError(ENDFRemoteError):
    """The isotope is not in the library, or its name cannot be read.

    Raised when the isotope string or number cannot be parsed, and when the server
    answers 404 for the file. For :func:`~kika.endf.download_endf` and
    :func:`~kika.endf.fetch_endf`, ``isotope`` and ``library`` are the values the
    caller passed; on the ``download_entry`` path they are the catalogue entry's
    label and library.
    """

    def __init__(self, isotope: str, library: str):
        self.isotope = isotope
        self.library = library
        super().__init__(f"Isotope '{isotope}' not found in library '{library}'")


class LibraryNotFoundError(ENDFRemoteError):
    """The library name is not one kika knows.

    ``library`` is the name the caller passed and ``available`` the keys of
    ``LIBRARY_PATHS``, as ``list_available_libraries()`` returns them.
    """

    def __init__(self, library: str, available: list[str]):
        self.library = library
        self.available = available
        super().__init__(
            f"Unknown library '{library}'. Available libraries: {', '.join(available)}"
        )


class NetworkError(ENDFRemoteError):
    """A request to the IAEA server failed.

    Raised on a timeout, an HTTP error other than 404 and a bot-protection
    refusal (see :class:`~kika.endf.remote.exceptions.AccessBlockedError`), a request that could not be made,
    and a ZIP archive from which the ENDF file could not be extracted. ``url`` is
    the address requested, when known.
    """

    def __init__(self, message: str, url: str | None = None):
        self.url = url
        super().__init__(message)


class AccessBlockedError(NetworkError):
    """Raised when the server answers but refuses automated clients.

    Raised when the answer is a 403, 429 or 503 carrying ``cf-mitigated:
    challenge`` or a Cloudflare "Just a moment..." page. The IAEA put
    ``nds.iaea.org`` behind such a managed challenge; a request that sends the
    allowlisted ``IAEA_USER_AGENT`` is let through. Retrying does not help; a web
    browser still gets through.
    """

    def __init__(self, url: str | None = None):
        super().__init__(
            "The server is refusing automated downloads (bot protection); "
            "the file can still be fetched from a web browser",
            url,
        )


class CacheError(ENDFRemoteError):
    """A file could not be written to the local download cache.

    Raised when writing a downloaded file into the cache fails with an
    ``OSError``, or when ``download_entry`` finds the file absent
    from the cache after downloading it.
    """

    pass
