"""Custom exceptions for ENDF remote download functionality."""


class ENDFRemoteError(Exception):
    """Base exception for ENDF remote operations."""

    pass


class IsotopeNotFoundError(ENDFRemoteError):
    """Raised when the requested isotope is not found in the library."""

    def __init__(self, isotope: str, library: str):
        self.isotope = isotope
        self.library = library
        super().__init__(f"Isotope '{isotope}' not found in library '{library}'")


class LibraryNotFoundError(ENDFRemoteError):
    """Raised when the requested library is not recognized."""

    def __init__(self, library: str, available: list[str]):
        self.library = library
        self.available = available
        super().__init__(
            f"Unknown library '{library}'. Available libraries: {', '.join(available)}"
        )


class NetworkError(ENDFRemoteError):
    """Raised when a network operation fails."""

    def __init__(self, message: str, url: str | None = None):
        self.url = url
        super().__init__(message)


class AccessBlockedError(NetworkError):
    """Raised when the server answers but refuses automated clients.

    The IAEA put ``nds.iaea.org`` behind a Cloudflare managed challenge (seen
    October 2026): every non-browser request gets a 403 carrying
    ``cf-mitigated: challenge`` and an HTML "Just a moment..." page, whatever
    its User-Agent. Retrying does not help; a web browser still gets through.
    """

    def __init__(self, url: str | None = None):
        super().__init__(
            "The server is refusing automated downloads (bot protection); "
            "the file can still be fetched from a web browser",
            url,
        )


class CacheError(ENDFRemoteError):
    """Raised when a cache operation fails."""

    pass
