"""Some helper functions."""

from pathlib import Path
from time import perf_counter
from typing import IO

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io.trajectory import TrajectoryWriter
from ase.mep import DimerControl, MinModeAtoms
from otfkmc.abc import hydra_parse

from graphatoms.enterpoint.config import Config
from graphatoms.system import Cluster, Gas  # type: ignore

from ._funcs import call_optimize, call_vib, run_dimer


class OptimizationFailed(RuntimeError):
    """Optimization failed."""


class CheckVibrationFailed(RuntimeError):
    """Check vibrations failed."""




def _angle_to_history(
    d_new: np.ndarray,
    history: dict[str, np.ndarray],
    thetacutoff: float,
) -> tuple[float, bool]:
    """Minimum angle (degrees) of ``d_new`` against any history vector.

    Pure function: flattens ``d_new`` and each history value to 1D, computes
    the cosine similarity ``dot/(|a|*|b|)`` clamped to ``[-1, 1]``, converts
    to degrees via ``arccos``, and returns
    ``(min_angle_degrees, should_skip)`` where ``should_skip`` is True iff the
    minimum angle is below ``thetacutoff`` (i.e. this probe is too close to a
    historical direction to bother re-running). Empty history returns
    ``(inf, False)``.

    Parameters
    ----------
    d_new : np.ndarray
        The candidate displacement vector, shape ``(3N,)`` or ``(N, 3)``.
    history : dict[str, np.ndarray]
        The historical displacement vectors (e.g. ``Scheduler.diffposition``).
    thetacutoff : float
        Skip threshold in degrees.

    Returns
    -------
    tuple[float, bool]
        ``(min_angle_degrees, should_skip)``.
    """
    if not history:
        return float("inf"), False
    new_flat = np.asarray(d_new, dtype=float).ravel()
    new_norm = np.linalg.norm(new_flat)
    if new_norm == 0.0:
        # zero vector is "trivially collinear"; skip to avoid divide-by-zero
        return 0.0, True
    angles: list[float] = []
    for v in history.values():
        v_flat = np.asarray(v, dtype=float).ravel()
        v_norm = np.linalg.norm(v_flat)
        if v_norm == 0.0:
            angles.append(0.0)
            continue
        cos = float(np.dot(new_flat, v_flat) / (new_norm * v_norm))
        cos = max(-1.0, min(1.0, cos))  # clamp for numerical safety
        # snap near-collinear cases to exactly 0/180 deg
        if cos >= 1.0 - 1e-12:
            angles.append(0.0)
        elif cos <= -1.0 + 1e-12:
            angles.append(180.0)
        else:
            angles.append(float(np.degrees(np.arccos(cos))))
    min_angle = min(angles)
    return min_angle, min_angle < thetacutoff


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
