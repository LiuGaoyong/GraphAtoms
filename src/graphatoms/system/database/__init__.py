"""The database for the system."""

from pathlib import Path

from .abc import DatabaseABC
from .dict import DictDB
from .dir import DirDB
from .hdf5 import AseH5DB
from .sqliteASE import AseSqliteDB


def get_db(
    format: str = "dict",
    path: Path | None = None,
    append: bool = False,
) -> DatabaseABC:
    """Get the database."""
    if format.lower() == "sqlite":
        assert path is not None, "The path must be not None."
        assert path.suffix == ".db", "The path must be a SQLite file."
        return AseSqliteDB(path, append=append)
    elif format.lower() == "dir":
        assert path is not None, "The path must be not None."
        return DirDB(path, append=append)
    elif format.lower() == "hdf5":
        assert path is not None, "The path must be not None."
        assert path.suffix == ".h5", "The path must be a HDF5 file."
        return AseH5DB(path, append=append)
    else:
        return DictDB()
