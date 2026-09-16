from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.config import Config
from graphatoms.reaction import Adsorption, Desorption, Reaction
from graphatoms.system import Cluster, Gas, SysGraph, System  # type: ignore
from graphatoms.utils import asetools
from graphatoms.utils.parser import hydra_parse

from ._errors import CheckVibrationFailed, HelperException, OptimizationFailed


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
        result_label = result.get_key_for_metadata()
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
            k = result.get_key_for_metadata()
            raise CheckVibrationFailed(
                frequencies=freq[:2],
                label=f"Init={result_label},TS={k}",
                fqmin=fqmin,
            )
    elif check_minima and (not check_ts):
        if not result.check_minima(fmax=config.event.max_force, fqmin=fqmin):
            k = result.get_key_for_metadata()
            raise CheckVibrationFailed(
                frequencies=freq[:1],
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
    raise_when_fail: bool = False,
    run_vibration: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Cluster | Gas | System | SysGraph | str, Any, float]:
    """Optimize the input graph, analyze its vibrations and check it.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        graph: The graph to optimization.
        allow_hash_change: Whether to allow the hash change after optimization.
        raise_when_fail: Whether to raise exception after optimization failed.
        run_vibration: Whether to run the vibration check after optimization.
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
        graph_label = graph.get_key_for_metadata()
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
        cost_time = perf_counter() - start
        e = OptimizationFailed(
            type="minima",
            max_steps=max(int(config.optimizer.steps), len(lst)),
            label=graph_label,
        )
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, cost_time

    # ---------------------------------------------
    #       check graph hash changed or not
    # ---------------------------------------------
    result = graph.update_geometry(
        new_positions=lst[-1].get_positions(),
        parse_bonds=config.bonds,  # type: ignore
        deep=deep_copy,
    )
    if not allow_hash_change and graph.hash != result.hash:
        e = HelperException(
            msg="hash changed after optimization",
            label=graph_label,
        )
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, perf_counter() - start
    if not run_vibration:
        return result, graph_label, perf_counter() - start

    # ---------------------------------------------
    #       call vibration & return result
    # ---------------------------------------------
    try:
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
    except CheckVibrationFailed as e:
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, perf_counter() - start


def helper_dimer(
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_fixed_bonds_change: bool = False,
    displacement: np.ndarray | None = None,
    raise_when_fail: bool = False,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Reaction | Desorption | str, Any, float]:
    """Dimer search for transition state.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        allow_fixed_bonds_change: Whether to
            allow the fixed bonds change after dimer.
        displacement: The displacement vector for dimer.
        raise_when_fail: Whether to raise exception when failed.
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
        graph_label = graph.get_key_for_metadata()
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
        cost_time = perf_counter() - start
        e = OptimizationFailed(
            type="dimer",
            label=graph_label,
            max_steps=max(int(config.optimizer.steps * 1.5), len(dimer_lst)),
        )
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, cost_time

    # -----------------------------------------
    #       check the fixed bond change or not
    # -----------------------------------------
    ts = graph.update_geometry(
        dimer_lst[-1].positions,
        parse_bonds=config.bonds,  # type: ignore
        deep=deep_copy,
    )
    k = ts.get_key_for_metadata()
    if not allow_fixed_bonds_change:
        break_bonds, make_bonds = graph.bond_difference(ts)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, graph.idx_fix)):
            cost_time = perf_counter() - start
            e = HelperException(
                msg="fixed bonds modified after dimer",
                label=f"{graph_label},TS={k}",
            )
            if raise_when_fail:
                raise e
            else:
                return str(e), graph_label, cost_time

    # -----------------------------------------
    #       analyze frequencies & check
    # -----------------------------------------
    try:
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
    except CheckVibrationFailed as e:
        cost_time = perf_counter() - start
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, cost_time

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
    opt_lst: list[Atoms] = []
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
        cost_time = perf_counter() - start
        e = OptimizationFailed(
            type="product",
            label=graph_label,
            max_steps=max(int(config.optimizer.steps), len(opt_lst)),
        )
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, cost_time

    # ------------------------------------------------------------
    #       check the fixed bond change or not for product
    # ------------------------------------------------------------
    if not product_result.is_connected:
        cost_time = perf_counter() - start
        e = HelperException(
            msg="product is not connected",
            label=graph_label,
        )
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, cost_time
    if not allow_fixed_bonds_change:
        break_bonds, make_bonds = graph.bond_difference(ts)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, graph.idx_fix)):
            kp = product_result.get_key_for_metadata()
            cost_time = perf_counter() - start
            e = HelperException(
                msg="fixed bonds modified for product",
                label=f"{graph_label},TS={k},P={kp}",
            )
            if raise_when_fail:
                raise e
            else:
                return str(e), graph_label, cost_time

    # -----------------------------------------
    # call vibration for product
    # -----------------------------------------
    try:
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
    except CheckVibrationFailed as e:
        if raise_when_fail:
            raise e
        else:
            return str(e), graph_label, perf_counter() - start


def helper_adsorption(
    gas: Gas,
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_fixed_bonds_change: bool = False,
    displacement: np.ndarray | None = None,
    raise_when_fail: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Adsorption | str, Any, float]:
    raise NotImplementedError
