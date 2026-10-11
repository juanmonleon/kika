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
# locally; on Read the Docs every version except the published "latest" and
# "stable" is a review build (pull-request previews, "develop", any review
# branch the maintainer activates).
_RTD_VERSION = os.environ.get("READTHEDOCS_VERSION")
BUILD_DRAFTS = (
    os.environ.get("KIKA_DOCS_DRAFTS") == "1"
    or (_RTD_VERSION is not None and _RTD_VERSION not in ("latest", "stable"))
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
# The PDF is the manual of methods and architecture: its root is pdf_index.rst,
# which leaves out the user guide and api_reference/ (both read online;
# docs/WRITING_DOCS.md §5.6). One document, built from that root.
# The title page has no room for the email beside the name.
latex_documents = [
    ('pdf_index', 'kika.tex', 'KIKA Documentation',
     'Juan Antonio Monleon de la Lluvia', 'manual'),
]
# An inline literal can only break after . , ; ? ! / by default, so a test
# reference such as file.py::test_a_long_name ran off the page. Let it break
# after : and _ as well. A code line with nowhere to break is cut anyway.
latex_elements = {
    'sphinxsetup': 'verbatimforcewraps=true',
    'preamble': r'''
\makeatletter
\g@addto@macro\sphinxbreaksafteractivelist{\do\:}
\g@addto@macro\sphinxbreaksviaactive{\def\_{\textunderscore\allowbreak}}
\makeatother
''',
}

# -- EPUB (also built by Read the Docs) ----------------------------------------
# A figure's PDF twin (for LaTeX) and the script that draws it live next to it
# under _static/ (docs/WRITING_DOCS.md §5.5). An EPUB has no media type for
# either, and says so as a warning, so they are left out of it; the EPUB uses
# the SVG.
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")
epub_exclude_files = sorted(
    os.path.relpath(os.path.join(root, name), os.path.dirname(_STATIC))
    for root, _dirs, names in os.walk(_STATIC)
    for name in names if name.endswith((".pdf", ".py"))
)


# -- Drafts wiring -----------------------------------------------------------

# The architecture overview opens its section; everything else follows by name.
_DRAFTS_TOCTREE = """

.. toctree::
   :maxdepth: 2
   :caption: Drafts (under review)
   :glob:

   drafts/architecture/index
   drafts/architecture/**
   drafts/methods/index
   drafts/methods/**
   drafts/user_guide/**
"""

_DRAFT_BANNER = """
```{warning}
**Draft under review.** This page is not part of the published manual yet, and
it may change before it is.
```
"""


def _wireDrafts(app, docname, source):
    """Add the drafts section to the root toctree, and the banner to each draft."""
    if docname == "index":
        source[0] += _DRAFTS_TOCTREE
    elif docname == "pdf_index":
        # The PDF is the manual of methods and architecture (WRITING_DOCS §5.6):
        # the user-guide drafts stay online.
        source[0] += _DRAFTS_TOCTREE.replace("   drafts/user_guide/**\n", "")
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
