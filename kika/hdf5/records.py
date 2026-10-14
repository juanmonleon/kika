"""The HDF5 tree as plain records, before any physics.

A file becomes a :class:`Group` of :class:`Group` and :class:`Dataset` children,
each with its attributes, and goes back to disk from the same records. Nothing
is interpreted here, in the way :mod:`kika.g4ndl.records` interprets nothing: a
fixed-length byte string stays a byte string (``b"data_neutron"``, not
``"data_neutron"``), a 1-based breakpoint stays 1-based, an ``int64`` stays an
``int64``. The decoders above this layer decide what each node means; this one
only guarantees that read → write gives back the same datasets, the same dtypes
and the same attributes.

What is *not* kept is the container: chunking, compression and the HDF5 library
version an object was written with are properties of the file, not of the data.
They are recorded on :class:`Dataset` when present so a copy can reproduce them
if asked, and :func:`compareTrees` ignores them.

**Lazy reading.** A thermal file of the official libraries weighs 2.1 GB, so a
tree can be opened lazily with :func:`openFile`: datasets read their values on
first access and :func:`writeFile` streams them one at a time. :func:`readFile`
loads everything and closes the file.

h5py is imported inside the functions, never at module scope, so that importing
:mod:`kika.hdf5` costs nothing until a file is touched.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

__all__ = ["Attribute", "Dataset", "Group", "Node", "readFile", "openFile",
           "writeFile", "compareTrees", "compareFiles", "walk", "attr", "text"]


@dataclass(eq=False)
class Attribute:
    """One HDF5 attribute: its value, exactly as stored, and its dtype and shape.

    ``value`` is what h5py hands back (``np.bytes_``, ``np.int64``, an array).
    The dtype is kept beside it because h5py's default on write is not the dtype
    the file had: a Python ``int`` would become ``int64`` everywhere, but a
    ``bytes`` of length 12 must come back as ``|S12`` and not ``|S13``.
    """

    value: object
    dtype: np.dtype
    shape: Tuple[int, ...]

    @classmethod
    def of(cls, value, dtype=None) -> "Attribute":
        """An attribute from a Python value, with the dtype OpenMC writes for it.

        ``str`` becomes a fixed-length byte string, ``int`` an ``int64`` and
        ``float`` a ``float64`` -- what ``h5py`` produces from the ``np.bytes_``,
        ``int`` and ``float`` values OpenMC's Python passes it.
        """
        pythonInts = isinstance(value, int) or (
            isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value))
        if isinstance(value, str):
            value = np.bytes_(value.encode())
        array = np.asarray(value, dtype=dtype)
        if array.dtype.kind == "U":
            array = array.astype("S")
        if dtype is None and pythonInts and array.dtype.kind == "i":
            array = array.astype(np.int64)
        stored = array[()] if array.ndim == 0 else array
        return cls(value=stored, dtype=array.dtype, shape=array.shape)


@dataclass(eq=False)
class Dataset:
    """One HDF5 dataset: its values, its attributes, and how it was stored."""

    attrs: Dict[str, Attribute] = field(default_factory=dict)
    _data: Optional[np.ndarray] = None
    #: The open ``h5py.Dataset`` a lazy record reads from, when there is one.
    _source: Optional[object] = None
    dtype: Optional[np.dtype] = None
    shape: Optional[Tuple[int, ...]] = None
    chunks: Optional[Tuple[int, ...]] = None
    compression: Optional[str] = None
    compression_opts: Optional[object] = None
    shuffle: bool = False

    @classmethod
    def of(cls, data, attrs: Optional[Dict[str, object]] = None, dtype=None) -> "Dataset":
        """A dataset from an array and plain attribute values."""
        array = np.asarray(data, dtype=dtype)
        record = cls(_data=array, dtype=array.dtype, shape=array.shape)
        for name, value in (attrs or {}).items():
            record.attrs[name] = value if isinstance(value, Attribute) else Attribute.of(value)
        return record

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            if self._source is None:
                raise ValueError("this dataset holds no values and no file to read them from")
            return self._source[()]
        return self._data

    @data.setter
    def data(self, value) -> None:
        array = np.asarray(value)
        self._data, self._source = array, None
        self.dtype, self.shape = array.dtype, array.shape

    def __repr__(self) -> str:
        return f"Dataset({self.dtype}{self.shape}, attrs={sorted(self.attrs)})"


@dataclass(eq=False)
class Group:
    """One HDF5 group: attributes and named children, in name order."""

    attrs: Dict[str, Attribute] = field(default_factory=dict)
    children: Dict[str, "Node"] = field(default_factory=dict)

    def __getitem__(self, path: str) -> "Node":
        node: Node = self
        for part in path.strip("/").split("/"):
            if not part:
                continue
            if not isinstance(node, Group):
                raise KeyError(path)
            node = node.children[part]
        return node

    def __contains__(self, path: str) -> bool:
        try:
            self[path]
        except KeyError:
            return False
        return True

    def __setitem__(self, name: str, node: "Node") -> None:
        if "/" in name.strip("/"):
            head, _, tail = name.strip("/").rpartition("/")
            self.require(head).children[tail] = node
        else:
            self.children[name.strip("/")] = node

    def require(self, path: str) -> "Group":
        """The group at ``path``, created along the way if absent."""
        node = self
        for part in path.strip("/").split("/"):
            if not part:
                continue
            child = node.children.get(part)
            if child is None:
                child = node.children[part] = Group()
            if not isinstance(child, Group):
                raise KeyError(f"{path}: {part} is a dataset, not a group")
            node = child
        return node

    def groups(self) -> Iterator[Tuple[str, "Group"]]:
        for name, child in self.children.items():
            if isinstance(child, Group):
                yield name, child

    def datasets(self) -> Iterator[Tuple[str, Dataset]]:
        for name, child in self.children.items():
            if isinstance(child, Dataset):
                yield name, child

    def setAttr(self, name: str, value, dtype=None) -> None:
        self.attrs[name] = value if isinstance(value, Attribute) else Attribute.of(value, dtype)

    def __repr__(self) -> str:
        return f"Group({len(self.children)} children, attrs={sorted(self.attrs)})"


Node = Union[Group, Dataset]


def attr(node: Node, name: str, default=None):
    """An attribute's value, or ``default`` when absent."""
    found = node.attrs.get(name)
    return default if found is None else found.value


def text(value) -> str:
    """A fixed-length HDF5 byte string as ``str``."""
    if isinstance(value, Attribute):
        value = value.value
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def _readAttrs(obj) -> Dict[str, Attribute]:
    out = {}
    for name in obj.attrs:
        aid = obj.attrs.get_id(name)
        out[name] = Attribute(value=obj.attrs[name], dtype=aid.dtype, shape=aid.shape)
    return out


def _readNode(obj, lazy: bool) -> Node:
    import h5py

    if isinstance(obj, h5py.Dataset):
        record = Dataset(attrs=_readAttrs(obj), dtype=obj.dtype, shape=obj.shape,
                         chunks=obj.chunks, compression=obj.compression,
                         compression_opts=obj.compression_opts, shuffle=obj.shuffle)
        if lazy:
            record._source = obj
        else:
            record._data = obj[()]
        return record
    group = Group(attrs=_readAttrs(obj))
    for name in obj:
        child = obj.get(name, getlink=True)
        if isinstance(child, (h5py.SoftLink, h5py.ExternalLink)):
            raise NotImplementedError(f"{obj.name}/{name} is an HDF5 link; OpenMC files hold none")
        group.children[name] = _readNode(obj[name], lazy)
    return group


def readFile(path) -> Group:
    """The whole file as records, loaded into memory. The file is closed on return."""
    import h5py

    with h5py.File(path, "r") as handle:
        return _readNode(handle, lazy=False)


@contextlib.contextmanager
def openFile(path):
    """The file as lazy records, valid inside the ``with`` block.

    Datasets read their values when ``.data`` is first touched and do not keep
    them, so walking a 2 GB thermal file costs one dataset of memory at a time.
    """
    import h5py

    handle = h5py.File(path, "r")
    try:
        yield _readNode(handle, lazy=True)
    finally:
        handle.close()


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def _writeAttrs(obj, attrs: Dict[str, Attribute]) -> None:
    for name, attribute in attrs.items():
        obj.attrs.create(name, attribute.value, shape=attribute.shape, dtype=attribute.dtype)


def _writeNode(parent, name: str, node: Node, keepStorage: bool) -> None:
    if isinstance(node, Dataset):
        data = node.data
        kwargs = {}
        if keepStorage and node.chunks is not None:
            kwargs = dict(chunks=node.chunks, compression=node.compression,
                          compression_opts=node.compression_opts, shuffle=node.shuffle)
        dataset = parent.create_dataset(name, data=data, dtype=node.dtype or data.dtype, **kwargs)
        _writeAttrs(dataset, node.attrs)
        return
    group = parent.create_group(name)
    _writeAttrs(group, node.attrs)
    for childName, child in node.children.items():
        _writeNode(group, childName, child, keepStorage)


def writeFile(root: Group, path, *, keepStorage: bool = False,
              libver: str = "earliest") -> Path:
    """Write the records to ``path``, replacing it.

    ``libver='earliest'`` is what OpenMC's exporter uses (``neutron.py:355``), so
    the files are readable by the oldest HDF5 the C++ may be linked against.
    ``keepStorage`` reproduces the chunking and compression a record was read
    with; by default every dataset is written contiguous and uncompressed, as
    the official libraries are.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver=libver) as handle:
        _writeAttrs(handle, root.attrs)
        for name, child in root.children.items():
            _writeNode(handle, name, child, keepStorage)
    return path


# ----------------------------------------------------------------------
# Comparing
# ----------------------------------------------------------------------

def walk(node: Node, prefix: str = "") -> Iterator[Tuple[str, Node]]:
    """``(path, node)`` for every node below ``node``, depth first, in name order."""
    yield prefix or "/", node
    if isinstance(node, Group):
        for name in sorted(node.children):
            yield from walk(node.children[name], f"{prefix}/{name}")


def _sameValue(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f" or a.dtype.kind == "c":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _compareAttrs(path: str, a: Dict[str, Attribute], b: Dict[str, Attribute],
                  out: List[str]) -> None:
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            out.append(f"{path} @{name}: only in {'first' if name in a else 'second'}")
            continue
        x, y = a[name], b[name]
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            out.append(f"{path} @{name}: {x.dtype}{tuple(x.shape)} != {y.dtype}{tuple(y.shape)}")
        elif not _sameValue(x.value, y.value):
            out.append(f"{path} @{name}: {x.value!r} != {y.value!r}")


def compareTrees(a: Node, b: Node, path: str = "", limit: Optional[int] = None) -> List[str]:
    """Every difference in data, dtype, shape and attributes, as readable lines.

    Empty means identical dataset by dataset. Storage --
    chunking, compression -- is not compared.
    """
    out: List[str] = []
    _compare(a, b, path, out, limit)
    return out


def _compare(a: Node, b: Node, path: str, out: List[str], limit) -> None:
    if limit is not None and len(out) >= limit:
        return
    where = path or "/"
    if type(a) is not type(b):
        out.append(f"{where}: {type(a).__name__} != {type(b).__name__}")
        return
    _compareAttrs(where, a.attrs, b.attrs, out)
    if isinstance(a, Dataset):
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            out.append(f"{where}: {a.dtype}{tuple(a.shape)} != {b.dtype}{tuple(b.shape)}")
        elif not _sameValue(a.data, b.data):
            out.append(f"{where}: values differ")
        return
    for name in sorted(set(a.children) | set(b.children)):
        if name not in a.children or name not in b.children:
            out.append(f"{path}/{name}: only in {'first' if name in a.children else 'second'}")
            continue
        _compare(a.children[name], b.children[name], f"{path}/{name}", out, limit)


def compareFiles(first, second, limit: Optional[int] = None) -> List[str]:
    """:func:`compareTrees` on two files, read lazily."""
    with openFile(first) as a, openFile(second) as b:
        return compareTrees(a, b, limit=limit)
