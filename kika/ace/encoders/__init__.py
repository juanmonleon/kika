"""Objects of an ``Ace`` back to XSS words: the inverse of ``kika.ace.parsers``.

``layout.encode_xss(ace)`` writes a neutron table's XSS, NXS and JXS from its
block objects in NJOY's layout, and ``layout.rebuild(ace)`` reads the result
back into a new ``Ace``: an ACE table can then be edited in structure (a
point added, a reaction removed), not only in value. ``blocks`` holds one
encoder per block parser and ``laws`` one per energy-law parser; ``_words``
is the buffer they write into.
"""
