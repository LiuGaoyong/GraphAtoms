import pathlib
from typing import Annotated

import pandas as pd
import typer
from ase import Atoms
from ase.visualize import view as _ase_view

from graphatoms.enterpoint.network import ReactionNetwork as Net
from graphatoms.system import SysGraph

__p_help__ = "Path to the network directory."
Path = Annotated[str, typer.Option("--path", "-p", help=__p_help__)]
pd.set_option("display.max_colwidth", None)  # don't truncate cell contents
pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.max_rows", None)  # show all rows
pd.set_option("display.width", None)  # don't wrap lines


# --------------
# Main command
# --------------
main = typer.Typer(no_args_is_help=True)


@main.command()
def summary(path: Path = ".") -> None:
    assert pathlib.Path(path).exists()
    net = Net(path, restart=True, format="dir")
    print(net.summary())


@main.command(no_args_is_help=True)
def merge(
    args: Annotated[
        list[str],
        typer.Argument(
            help=f"{Net.__name__} directories to merge (one or more)"
        ),
    ],
    path: Path = "./event",
    db_format: Annotated[
        str,
        typer.Option(
            "--db-format",
            "-f",
            help="Database format in the network.",
        ),
    ] = "dir",
) -> None:
    assert db_format == "dir", "Only `dir` format is supported now."
    lst = [Net(p, restart=True, format=db_format) for p in args]
    Net.merge(*lst, new_path=path)


# ----------------
# Table subcommand
# ----------------
show = typer.Typer(no_args_is_help=True)
main.add_typer(show, name="show")


@show.command(name="reaction")
def show_reaction(arg: str, path: Path = ".") -> tuple[int, Net]:
    net = Net(path, restart=True, format="dir")
    df = net.metadata.table.dataframe
    try:
        i = int(arg)
    except ValueError:
        i = net.metadata.table.key_rxn.index(arg)
    print(df.iloc[i])
    return i, net


@show.command()
def table(path: Path = ".") -> None:
    assert pathlib.Path(path).exists()
    net = Net(path, restart=True, format="dir")
    df = net.metadata.table.dataframe
    col = [0, 5, 6, 7, 8, 11, 12]
    print(df[df.columns[col]])


# ----------------
# View subcommand
# ----------------
view = typer.Typer(no_args_is_help=True)

main.add_typer(view, name="view")


@view.command(name="reaction")
def view_reaction(
    name: str,
    path: Path = ".",
    forward: bool = True,
) -> None:
    idx, net = show_reaction(name, path)
    key_rxn = net.metadata.table.key_rxn[idx]
    _, rxn = net.read_event(key_rxn)
    if not forward:
        rxn = rxn.reversed
    return rxn.view()


@view.command(name="structure")
def view_structure(
    name_or_path: str,
    path: Path = ".",
) -> None:
    try:
        fname = pathlib.Path(name_or_path)
        if fname.name.count("*") == 0:
            assert fname.exists() and fname.is_file()
            return _ase_view(SysGraph.read_npz(fname).to_ase())
        else:
            return _ase_view(
                [
                    SysGraph.read_npz(i).to_ase()
                    for i in fname.parent.glob(fname.name)
                ]
            )
    except ValueError:
        name = name_or_path

    assert pathlib.Path(path).exists()
    net = Net(path, restart=True, format="dir")
    atoms: Atoms | None = None
    for type in ("gas", "ts", "minima", "cluster", "system"):
        if net.contain_sysgraph(name, type=type):
            atoms = net.read_atoms(name, type=type)
            break
    if atoms is None:
        raise ValueError(f"structure {name} is not in the database.")
    return _ase_view(atoms)


if __name__ == "__main__":
    main()
