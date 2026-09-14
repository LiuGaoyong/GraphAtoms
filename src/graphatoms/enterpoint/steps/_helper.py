from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.config import Config
from graphatoms.reaction import Adsorption, Desorption, Reaction
from graphatoms.system import Cluster, Gas, SysGraph, System  # type: ignore
from graphatoms.system.database import DatabaseABC
from graphatoms.utils import asetools
from graphatoms.utils.parser import hydra_parse


class HelperException(Exception):
    def __init__(
        self,
        *,
        msg: str,
        cost_time: float | None = None,
        label: str | None = None,
    ) -> None:
        cost = f"CostTime={cost_time:.2f}. " if cost_time is not None else ""
        label = f" for ({label})" if label is not None else ""
        super().__init__(cost + msg + label)


class OptimizationFailed(HelperException):
    def __init__(
        self,
        *,
        type: str,
        cost_time: float | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(
            msg=f"{type} optimization not coveraged",
            cost_time=cost_time,
            label=label,
        )


class CheckVibrationFailed(HelperException):
    def __init__(
        self,
        *,
        fqmin: float | None = None,
        frequencies: np.ndarray | None = None,
        cost_time: float | None = None,
        label: str | None = None,
    ) -> None:
        if frequencies is None:
            fstr = ""
        else:
            fstr = ",".join(f"{f:.2f}" for f in frequencies[:3])
            if fqmin is None:
                fstr = f"({fstr})"
            else:
                fstr = f"({fstr}|FQMIN={fqmin:.2f})"
        super().__init__(
            msg=f"check frequencies{fstr} failed",
            cost_time=cost_time,
            label=label,
        )


def __helper_vibration(
    config: Config,
    result: Cluster | Gas | System | SysGraph,
    *,
    check_ts: bool = False,
    check_minima: bool = True,
    result_label: Any | None = None,
    result_atoms: Atoms | None = None,
    start: float | None = None,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Cluster | Gas | System | SysGraph, np.ndarray]:
    if start is None:
        start = perf_counter()
    if result_label is None:
        result_label = DatabaseABC.get_key_of(result)
    if result_atoms is None:
        result_atoms = result.to_ase(exclude_bond_attibutes=True).copy()
    else:
        result_atoms = result_atoms.copy()
    assert isinstance(result_atoms, Atoms)
    calc: Calculator = hydra_parse(
        config.calculator,  # type: ignore
        Calculator,
    )

    if check_ts and check_minima:
        raise ValueError("check_ts and check_minima cannot be both True")
    elif check_ts and (not check_minima):
        fqmin = float(config.event.min_frequency_for_ts)
    elif check_minima and (not check_ts):
        fqmin = float(config.event.min_frequency)
    else:
        fqmin = 1.0
    freq, modes = asetools.call_vib(result_atoms, calc, ignore_fqmin=fqmin)
    f: np.ndarray = result_atoms.get_forces(apply_constraint=True)
    result = result.update_energetics(
        energy=result_atoms.get_potential_energy(),
        fmax=np.linalg.norm(f, axis=1).max(),
        frequencies=freq,
        deep=deep_copy,
    )

    if check_ts and check_minima:
        raise ValueError("check_ts and check_minima cannot be both True")
    elif check_ts and (not check_minima):
        if not result.check_ts(fmax=float(config.event.max_force), fqmin=fqmin):
            k = DatabaseABC.get_key_of(result)
            raise CheckVibrationFailed(
                frequencies=freq,
                cost_time=perf_counter() - start,
                label=f"Init={result_label},TS={k}",
                fqmin=fqmin,
            )
    elif check_minima and (not check_ts):
        if not result.check_minima(fmax=config.event.max_force, fqmin=fqmin):
            k = DatabaseABC.get_key_of(result)
            raise CheckVibrationFailed(
                frequencies=freq,
                cost_time=perf_counter() - start,
                label=f"Init={result_label},Minima={k}",
                fqmin=fqmin,
            )

    return result, modes


def helper_optimization(
    config: Config,
    graph: Cluster | Gas | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_hash_change: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Cluster | Gas | System | SysGraph, Any, float]:
    """Optimize the input graph, analyze its vibrations and check it.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        graph: The graph to optimization.
        allow_hash_change: Whether to allow the hash change after optimization.
        deep_copy: Whether to deep copy the optimized graph.

    Raises:
        OptimizationFailed: If the optimization failed.
        CheckVibrationFailed: If the check of vibration failed.
        HelperException: If the graph hash changed after optimization.

    Returns:
        the optimized graph, graph_id, and the time cost in seconds.
    """
    start = perf_counter()
    if graph_label is None:
        graph_label = DatabaseABC.get_key_of(graph)
    calc: Calculator = hydra_parse(
        config.calculator,  # type: ignore
        Calculator,
    )

    # ---------------------------------------------
    #       call optimization
    # ---------------------------------------------
    lst, coveraged = asetools.call_optimization(
        atoms=graph.to_ase(
            exclude_bond_attibutes=True,
            exclude_energy=True,
        ).copy(),
        calc=calc,
        method=str(config.optimizer.method).upper(),
        max_steps=int(config.optimizer.steps),
        fmax=float(config.optimizer.fmax),
    )
    if not coveraged:
        raise OptimizationFailed(
            type="dimer",
            label=graph_label,
            cost_time=perf_counter() - start,
        )

    # ---------------------------------------------
    #       check graph hash changed or not
    # ---------------------------------------------
    result = graph.update_geometry(
        new_positions=lst[-1].get_positions(),
        parse_bonds=config.bonds,  # type: ignore
        deep=deep_copy,
    )
    if not allow_hash_change and graph.hash != result.hash:
        raise HelperException(
            msg="hash changed after optimization",
            cost_time=perf_counter() - start,
            label=graph_label,
        )

    result, _ = __helper_vibration(
        result_atoms=lst[-1].copy(),
        result_label=graph_label,
        deep_copy=deep_copy,
        check_minima=True,
        check_ts=False,
        config=config,
        result=result,
        start=start,
    )
    return result, graph_label, perf_counter() - start


def helper_dimer(
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_fixed_bonds_change: bool = False,
    displacement: np.ndarray | None = None,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Reaction | Desorption, Any, float]:
    """Dimer search for transition state.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        allow_fixed_bonds_change: Whether to
            allow the fixed bonds change after dimer.
        displacement: The displacement vector for dimer.
        deep_copy: Whether to deep copy the optimized graph.

    Raises:
        OptimizationFailed: If the dimer failed.
        CheckVibrationFailed: If the check of vibration failed.
        HelperException: If the fixed bonds changed after dimer search.

    Returns:
        the optimized graph, graph_id, and the time cost in seconds.
    """

    start = perf_counter()
    if graph_label is None:
        graph_label = DatabaseABC.get_key_of(graph)
    calc: Calculator = hydra_parse(
        config.calculator,  # type: ignore
        Calculator,
    )

    # -----------------------------------------
    #       call dimer for TS search
    # -----------------------------------------
    mask: np.ndarray = np.ones(graph.natoms, dtype=bool)
    if len(graph.idx_fix) != 0:  # patch for the fixed bonds
        graph_dist = graph.get_hop_distance(graph.idx_fix)
        mask[graph_dist <= 1] = False
    dimer_lst, coveraged = asetools.call_dimer(
        atoms=graph.to_ase(
            exclude_bond_attibutes=True,
            exclude_energetics=True,
        ).copy(),
        calc=calc,
        # logfile="-",
        # trajectory="dimer.traj",
        append_trajectory=False,
        parse_mask_from_atoms=True,
        max_steps=int(config.optimizer.steps * 1.5),
        fmax=float(config.event.max_force) * 0.5,
        # Eecause dimer use projected force as
        # criterion, so we use smaller force
        # threshold to ensure check TS good.
        displacement=displacement,
        mask=mask,
        **kwargs,
    )
    if not coveraged:
        raise OptimizationFailed(
            type="dimer",
            label=graph_label,
            cost_time=perf_counter() - start,
        )

    # -----------------------------------------
    #       check the fixed bond change or not
    # -----------------------------------------
    ts = graph.update_geometry(
        dimer_lst[-1].positions,
        parse_bonds=config.bonds,  # type: ignore
        deep=deep_copy,
    )
    k = DatabaseABC.get_key_of(ts)
    if not allow_fixed_bonds_change:
        break_bonds, make_bonds = graph.bond_difference(ts)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, graph.idx_fix)):
            raise HelperException(
                msg="fixed bonds modified after dimer",
                cost_time=perf_counter() - start,
                label=f"{graph_label},TS={k}",
            )

    # -----------------------------------------
    #       analyze frequencies & check
    # -----------------------------------------
    ts, vib_modes = __helper_vibration(
        result_atoms=dimer_lst[-1].copy(),
        result_label=graph_label,
        deep_copy=deep_copy,
        check_minima=False,
        check_ts=True,
        config=config,
        start=start,
        result=ts,
    )

    # -----------------------------------------
    #        descend the minimum mode
    # -----------------------------------------
    vdiff = dimer_lst[-1].positions - graph.positions
    ldiff = np.linalg.norm(vdiff, axis=1)
    ldiff[ldiff < 1e-5] = np.inf
    imin_ldiff = np.argmin(ldiff)
    lmin_mode = np.linalg.norm(vib_modes[0][imin_ldiff])
    mode = vib_modes[0] * vdiff[imin_ldiff] / lmin_mode
    product_result: SysGraph | None = None
    product_atoms: Atoms | None = None
    for sign in (1, -1):
        atoms = dimer_lst[-1].copy()
        atoms.info.pop("hashes", None)
        atoms.positions += sign * mode
        opt_lst, coveraged = asetools.call_optimization(
            atoms=atoms,
            calc=calc,
            method=str(config.optimizer.method).upper(),
            max_steps=int(config.optimizer.steps),
            fmax=float(config.optimizer.fmax),
            # trajectory=f"opt_{sign}.traj",
            # logfile="-",
        )
        if coveraged:
            product_result = graph.update_geometry(
                opt_lst[-1].positions,
                parse_bonds=config.bonds,  # type: ignore
                deep=deep_copy,
            )
            if product_result.hash != graph.hash:  # type: ignore
                product_atoms = opt_lst[-1]
                break
    if product_result is None or product_atoms is None:
        raise OptimizationFailed(
            type="product",
            label=graph_label,
            cost_time=perf_counter() - start,
        )

    # ------------------------------------------------------------
    #       check the fixed bond change or not for product
    # ------------------------------------------------------------
    if not product_result.is_connected:
        raise HelperException(
            msg="product is not connected",
            cost_time=perf_counter() - start,
            label=graph_label,
        )
    if not allow_fixed_bonds_change:
        break_bonds, make_bonds = graph.bond_difference(ts)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, graph.idx_fix)):
            kp = DatabaseABC.get_key_of(product_result)
            raise HelperException(
                msg="fixed bonds modified for product",
                cost_time=perf_counter() - start,
                label=f"{graph_label},TS={k},P={kp}",
            )

    # -----------------------------------------
    # call vibration for product
    # -----------------------------------------
    product_result, _ = __helper_vibration(
        result_atoms=product_atoms,
        result_label=graph_label,
        result=product_result,
        deep_copy=deep_copy,
        check_minima=True,
        check_ts=False,
        config=config,
        start=start,
    )

    rxn = Reaction(R=graph, P=product_result, T=ts)
    return rxn, graph_label, perf_counter() - start


def helper_adsorption() -> Adsorption:
    raise NotImplementedError
