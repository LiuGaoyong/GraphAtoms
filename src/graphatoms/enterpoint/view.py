import pathlib
from typing import Annotated

import typer
from ase import Atoms
from ase.visualize import view as _ase_view

from graphatoms.enterpoint.network import RxNet
from graphatoms.system import SysGraph

__p_help__ = "Path to the network directory."
Path = Annotated[str, typer.Option("--path", "-p", help=__p_help__)]


main = typer.Typer(no_args_is_help=True)
view = typer.Typer(no_args_is_help=True)
show = typer.Typer(no_args_is_help=True)

main.add_typer(view, name="view")
main.add_typer(show, name="show")


@show.command(no_args_is_help=True)
def show_rxn(path: Path) -> None:
    print(f"Show reaction {path}")


@view.command(no_args_is_help=True)
def reaction(
    name: str,
    path: Path = ".",
    forward: bool = True,
) -> None:
    assert pathlib.Path(path).exists()
    net = RxNet(path, restart=True, format="dir")
    _, rxn = net.read(name)
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


@view.command(no_args_is_help=True)
def structure(
    name: str,
    path: Path = ".",
) -> None:
    assert pathlib.Path(path).exists()
    net = RxNet(path, restart=True, format="dir")
    if name in net.metadata.table.key_g:
        atoms: Atoms = net.db_gas[name]
    elif name in net.metadata.table.key_t:
        atoms: Atoms = net.db_ts[name]
    elif name in net.metadata.table.key_r:
        atoms: Atoms = net.db_minima[name]
    elif name in net.metadata.table.key_p:
        atoms: Atoms = net.db_minima[name]
    else:
        raise ValueError(f"structure {name} is not in the database.")
    return _ase_view(atoms)


@main.command(no_args_is_help=True)
def summary(path: Path = ".") -> None:
    assert pathlib.Path(path).exists()
    net = RxNet(path, restart=True, format="dir")
    print(net.summary())


if __name__ == "__main__":
    main()
