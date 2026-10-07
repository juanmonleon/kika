"""Modeled GNDS documentation: texts, authors and dates (xData schema).

Other documentation collections are explicitly reported by the adapter; this
module does not store raw XML or guess missing provenance.
"""
from dataclasses import dataclass, field


@dataclass
class DocumentationText:
    text: str = ""
    encoding: str | None = None
    markup: str | None = None
    label: str | None = None


@dataclass
class Author:
    name: str
    orcid: str | None = None
    email: str | None = None


@dataclass
class Date:
    value: str
    dateType: str


@dataclass
class Documentation:
    doi: str | None = None
    publicationDate: str | None = None
    version: str | None = None
    title: DocumentationText | None = None
    abstract: DocumentationText | None = None
    body: DocumentationText | None = None
    endfCompatible: DocumentationText | None = None
    authors: list[Author] = field(default_factory=list)
    dates: list[Date] = field(default_factory=list)
