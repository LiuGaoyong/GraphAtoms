"""Some helper functions."""

from pathlib import Path
from time import perf_counter
from typing import IO

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io.trajectory import TrajectoryWriter
from ase.mep import DimerControl, MinModeAtoms

from graphatoms.enterpoint.config import Config
from graphatoms.system import Cluster  # type: igno


class OptimizationFailed(RuntimeError):
    """Optimization failed."""


class CheckVibrationFailed(RuntimeError):
    """Check vibrations failed."""


def _helper_dimer_ts(
    config: Config,
    cluster: Cluster,
    *,
    d_control: DimerControl,
    d_atoms: MinModeAtoms,
    data: list[Atoms],
    traj: TrajectoryWriter | None,
    max_steps: int,
    fmax: float,
    logfile: IO | Path | str | None = None,
    raise_on_failed: bool = False,
    **kwargs,
) -> tuple[Cluster | str, np.ndarray | None, float]:
    """Complete the dimer search (run_dimer) + TS-check tail.

    Consumes the pre-init dimer state from :func:`init_dimer` and completes
    the search via :func:`run_dimer` (rather than calling :func:`call_dimer`
    itself), so the pre-judged displacement is the one actually used. The vib
    + TS-check tail (``call_vib`` + ``Cluster.from_ase`` + ``check_ts``) is
    unchanged from the previous implementation. ``run_dimer`` closes
    ``d_control`` itself (its context exit), so callers do NOT need to clean
    up on the happy or failed path.

    Parameters
    ----------
    config, cluster : see existing implementation
    d_control, d_atoms, data, traj : from :func:`init_dimer`
    max_steps, fmax, logfile : forwarded to :func:`run_dimer`
    raise_on_failed : raise on failure instead of returning an error string

    Returns
    -------
    tuple[Cluster | str, np.ndarray | None, float]
        ``(ts_cluster | error_msg, modes | None, elapsed)``.
    """
    assert cluster.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    )
    start = perf_counter()
    calc: Calculator = hydra_parse(
        config.calculator,  # type: ignore
        Calculator,
    )

    # complete the dimer search (run_dimer closes d_control itself)
    lst, coveraged = run_dimer(
        d_control,
        d_atoms,
        data=data,
        traj=traj,
        max_steps=max_steps,
        fmax=fmax,
        logfile=logfile,
    )
    if not coveraged:
        msg = f"Dimer failed for {cluster}."
        if raise_on_failed:
            raise OptimizationFailed(msg)
        else:
            return msg, None, perf_counter() - start

    # convert the dimer result to a TS cluster & check it (unchanged physics)
    new_atoms = lst[-1]
    freq, modes = call_vib(atoms=new_atoms, calc=calc)
    f = new_atoms.get_forces()
    ts = cluster.from_ase(
        new_atoms,
        parse_bonds=config.bonds,  # type: ignore
        parse_bonds_distance=False,
        parse_bonds_order=False,
        energy=new_atoms.get_potential_energy(),
        fmax=np.linalg.norm(f, axis=1).max(),
        frequencies=freq,
        nadsorbate=0,
    )
    if not ts.check_ts(
        fmax=config.event.max_force,
        fqmin=config.event.min_frequency_for_ts,
    ):
        msg = f"Check TS failed for {ts}."
        if raise_on_failed:
            raise CheckVibrationFailed(msg)
        else:
            return msg, None, perf_counter() - start
    return ts, modes, perf_counter() - start


def _helper_dimer_descend_mode(
    config: Config,
    cluster: Cluster,
    transition: Cluster,
    modes: np.ndarray,
    *,
    raise_on_failed: bool = False,
    **kwargs,
) -> tuple[Cluster | str, float]:
    start = perf_counter()
    assert cluster.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    )
    assert transition.check_ts(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency_for_ts),
    )
    vdiff = transition.positions - cluster.positions
    ldiff = np.linalg.norm(vdiff, axis=1)
    ldiff[ldiff < 1e-5] = np.inf
    imin_ldiff = np.argmin(ldiff)
    lmin_mode = np.linalg.norm(modes[0][imin_ldiff])
    mode = modes[0] * vdiff[imin_ldiff] / lmin_mode

    for sign in (1, -1):
        atoms = transition.to_ase(
            exclude_bond_attibutes=True,
            exclude_energetics=True,
        ).copy()
        atoms.info.pop("hashes", None)
        atoms.positions += sign * mode
        try:
            cluster_optimized_0, _ = helper_cluster_optimization(
                config=config,
                cluster=cluster.from_ase(
                    atoms,
                    parse_bonds=config.bonds,  # type: ignore
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    energy=None,
                    fmax=None,
                    frequencies=None,
                    nadsorbate=0,
                ),
                raise_on_failed=True,
                allow_hash_change=True,
            )
            if cluster_optimized_0.hash != cluster.hash:  # type: ignore
                return cluster_optimized_0, perf_counter() - start
        except Exception:
            sign_str = "+" if sign > 0 else "-"
            msg = f"Optimization failed for {cluster} in direction {sign_str} ."
            if raise_on_failed:
                raise OptimizationFailed(msg)
            else:
                return msg, perf_counter() - start

    return "No optimized cluster found.", perf_counter() - start
