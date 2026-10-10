import datetime
import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'KIKA'
copyright = f"{datetime.datetime.now().year}, Juan Antonio Monleon de la Lluvia <juanmonleon96@gmail.com>"
author = "Juan Antonio Monleon de la Lluvia <juanmonleon96@gmail.com>"
# The installed distribution's version, so the site never states a stale one.
try:
    release = metadata.version("kika-nd")
except metadata.PackageNotFoundError:
    release = "unknown"
version = release

# -- Drafts ------------------------------------------------------------------
# docs/source/drafts/ holds pages the maintainer has not reviewed yet
# (docs/WRITING_DOCS.md §9). The published site (Read the Docs "latest" and
# "stable") leaves them out. A review build includes them, under their own
# "Drafts" section and with a banner on every page: set KIKA_DOCS_DRAFTS=1
# locally; on Read the Docs, pull-request previews and the "develop" version
# are review builds.
BUILD_DRAFTS = (
    os.environ.get("KIKA_DOCS_DRAFTS") == "1"
    or os.environ.get("READTHEDOCS_VERSION_TYPE") == "external"
    or os.environ.get("READTHEDOCS_VERSION") == "develop"
)

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# $...$ and $$...$$ math, and amsmath environments (docs/WRITING_DOCS.md §5.3).
myst_enable_extensions = ['dollarmath', 'amsmath']

templates_path = ['_templates']
exclude_patterns = [] if BUILD_DRAFTS else ['drafts']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
# An "Attributes" section becomes fields of its class, not separate objects:
# autodoc already documents a dataclass's fields as members, and two entries
# for one attribute is a duplicate-object warning.
napoleon_use_ivar = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': True,
}

# -- HTML output options -----------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3,
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_with_keys': True,
    'logo_only': False,
    'style_external_links': True,
    'includehidden': True,
}

html_static_path = ['_static']

html_title = "KIKA Documentation"
html_short_title = "KIKA"

# -- LaTeX (the PDF Read the Docs builds) -------------------------------------
# XeLaTeX reads the Unicode the pages and docstrings use (Greek, ℓ, ⁻, ·)
# directly; pdflatex stops at the first character it has no mapping for.
latex_engine = 'xelatex'


# -- Drafts wiring -----------------------------------------------------------

_DRAFTS_TOCTREE = """

.. toctree::
   :maxdepth: 2
   :caption: Drafts (under review)
   :glob:

   drafts/**
"""

_DRAFT_BANNER = """
```{warning}
**Draft under review.** This page is not part of the published manual yet, and
it may change before it is.
```
"""


def _wireDrafts(app, docname, source):
    """Add the drafts section to the root toctree, and the banner to each draft."""
    if docname == app.config.root_doc:
        source[0] += _DRAFTS_TOCTREE
    elif docname.startswith("drafts/"):
        lines = source[0].split("\n")
        title = next((i for i, line in enumerate(lines) if line.startswith("# ")), -1)
        lines.insert(title + 1, _DRAFT_BANNER)
        source[0] = "\n".join(lines)


def setup(app):
    # With no draft left, the glob would match nothing, and that is a warning.
    drafts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drafts")
    hasDrafts = any(name.endswith((".md", ".rst"))
                    for _root, _dirs, names in os.walk(drafts) for name in names)
    if BUILD_DRAFTS and hasDrafts:
        app.connect("source-read", _wireDrafts)
