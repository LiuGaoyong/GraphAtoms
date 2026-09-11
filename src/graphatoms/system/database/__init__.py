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
    prefix: str | None = None,
    append: bool = False,
) -> DatabaseABC:
    """Get the database."""
    if format.lower() == "sqlite":
        assert path is not None, "The path must be not None."
        if prefix is None:
            assert path.suffix == ".db", "The path must be a SQLite file."
        else:
            path = path.joinpath(f"{prefix}.db")
        return AseSqliteDB(path, append=append)
    elif format.lower() in ["folder", "directory", "dir"]:
        assert path is not None, "The path must be not None."
        if prefix is not None:
            path = path.joinpath(prefix)
        return DirDB(path, append=append)
    elif format.lower() in ["hdf5", "h5"]:
        assert path is not None, "The path must be not None."
        if prefix is None:
            assert path.suffix == ".h5", "The path must be a HDF5 file."
        else:
            path = path.joinpath(f"{prefix}.h5")
        return AseH5DB(path, append=append)
    else:
        return DictDB()
