import pathlib
from typing import Annotated

import typer
from ase import Atoms
from ase.visualize import view as _ase_view

from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.system import SysGraph

__p_help__ = "Path to the network directory."
Path = Annotated[str, typer.Option("--path", "-p", help=__p_help__)]

# --------------
# Main command
# --------------
main = typer.Typer(no_args_is_help=True)


@main.command()
def summary(path: Path = ".") -> None:
    assert pathlib.Path(path).exists()
    net = ReactionNetwork(path, restart=True, format="dir")
    print(net.summary())


@main.command()
def table(path: Path = ".") -> None:
    assert pathlib.Path(path).exists()
    net = ReactionNetwork(path, restart=True, format="dir")
    df = net.metadata.table.dataframe
    col = [0, 5, 6, 7, 8, 11, 12]
    print(df[df.columns[col]])


@main.command(no_args_is_help=True)
def merge(
    args: Annotated[
        list[str],
        typer.Argument(
            help="ReactionNetwork directories to merge (one or more)"
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
    lst = [ReactionNetwork(p, restart=True, format=db_format) for p in args]
    ReactionNetwork.merge(*lst, new_path=path)


# ----------------
# Table subcommand
# ----------------
show = typer.Typer(no_args_is_help=True)
main.add_typer(show, name="show")


@show.command()
def show_rxn(path: Path = ".") -> None:
    raise NotImplementedError()


# ----------------
# View subcommand
# ----------------
view = typer.Typer(no_args_is_help=True)

main.add_typer(view, name="view")


@view.command()
def reaction(
    name: str,
    path: Path = ".",
    forward: bool = True,
) -> None:
    assert pathlib.Path(path).exists()
    net = ReactionNetwork(path, restart=True, format="dir")
    _, rxn = net.read_event(name)
    if not forward:
        rxn = rxn.reversed
    names, lst = [], []
    for k in rxn.__pydantic_fields__:
        v: SysGraph = getattr(rxn, k)
        if v is not None:
            names.append(k)
            lst.append(v)

    return _ase_view([i.to_ase() for i in lst])
    try:
        from tempfile import TemporaryDirectory

        import chemiscope

        with TemporaryDirectory() as tmpdir:
            ppp = pathlib.Path(tmpdir) / "input.json"
            chemiscope.write_input(
                # ppp.as_posix(),
                "input.json",
                structures=[i.to_ase() for i in lst],
                properties={
                    "names": {
                        "target": "structure",
                        "values": names,
                        "units": "eV",
                        "description": "DFT total energy",
                    },
                    "energy": {
                        "target": "structure",
                        "values": [i.energy for i in lst],
                        "units": "eV",
                        "description": "DFT total energy",
                    },
                },
            )
            return chemiscope.show_input(ppp)
    except ImportError:
        return _ase_view([i.to_ase() for i in lst])


@view.command()
def structure(
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
    net = ReactionNetwork(path, restart=True, format="dir")
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
