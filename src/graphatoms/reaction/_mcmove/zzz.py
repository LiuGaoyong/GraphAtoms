import typer
from numpy import nan

from ._simNVT import MonteCarloNVT

mmc_app = typer.Typer()


@mmc_app.command()
def nvt(
    structure_file: str | None = typer.Option(
        None,
        "--structure-file",
        "-s",
        help="""The file to read the structure from.
            By default, the structure will be read
            from either 'POSCAR' or 'structure.xyz'.""",
    ),
    spe_method: str = typer.Option(
        "emt",
        "--spe-method",
        "-spe",
        help="""The SPE method for simulation.
            It's format like NAME,ARG0,ARG1,ARG...,KWARG0=VAL0,KWARG1=VAL1,...
        """,
    ),
    swap_for_all_pairs: bool = typer.Option(
        False,
        "--swap-for-all-pairs",
        help="""Wether perform atoms swap for all atomic
            pair or only for the connected atomic pairs.""",
    ),
    swap_interval: int = typer.Option(
        0,
        "--swap-interval",
        help="""Perform atoms swap for every *interval* time steps.
            If it was set to zero, no swap will be performed.
            """,
    ),
    record_interval: int = typer.Option(
        1,
        "--record-interval",
        help="""Only write record line for every *interval* time steps""",
    ),
    trajectory_interval: int = typer.Option(
        1,
        "--trajectory-interval",
        help="""Only write trajectory for every *interval* time steps""",
    ),
    trajectory_file: str = typer.Option(
        "ase.traj",
        "--trajectory-file",
        help="""The file to write the trajectory.""",
    ),
    record_file: str = typer.Option(
        "record.table",
        "--record-file",
        help="""The file to record something.
            Typically, the record file is a table of the simulation
            results and it can be read as a pandas dataframe.""",
    ),
    max_steps: int = typer.Option(
        100_000,
        "--max-steps",
        help="""The maximum number of steps for simulation.""",
    ),
    stepsize_max: float = typer.Option(
        0.5,
        "--stepsize-max",
        help="""The maximum stepsize for single particle displacement.""",
    ),
    temperature: float = typer.Option(
        1000.0,
        "--temperature",
        help="""The temperature for NVT simulation.""",
    ),
    lambda_for_force_bias: float = typer.Option(
        nan,
        "--lambda-for-force-bias",
        help="""The lambda parameter for force bias.""",
    ),
    workdir: str = typer.Option(
        ".",
        "--workdir",
        help="""The working directory.""",
    ),
) -> None:
    sim = MonteCarloNVT(
        atoms=structure_file,
        calculator=spe_method,
        record_file=record_file,
        trajectory=trajectory_file,
        trajectory_interval=trajectory_interval,
        record_interval=record_interval,
        workdir=workdir,
        stepsize_max=stepsize_max,
        temperature=temperature,
        swap_interval=swap_interval,
        swap_for_connected_pair_only=not swap_for_all_pairs,
        lambda_for_force_bias=lambda_for_force_bias,
        max_steps=max_steps,
    )
    sim.run()
