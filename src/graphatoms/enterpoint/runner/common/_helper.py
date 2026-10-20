"""The basic helper functions for exploration and optimization.

These helper functions only run calculation, and check the result.
They do not check whether the calculation is necessary and output logger.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io.extxyz import write_xyz as write_extxyz

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.reaction import Adsorption, Desorption, Reaction
from graphatoms.system import Cluster, Gas, SysGraph, System
from graphatoms.system.atoms import Structure
from graphatoms.utils import asetools
from graphatoms.utils.adsorption import Helper
from graphatoms.utils.bytestool import atomic_write
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
                label=f"Init={result_label}, TS={k}",
                fqmin=fqmin,
            )
    elif check_minima and (not check_ts):
        if not result.check_minima(fmax=config.event.max_force, fqmin=fqmin):
            k = result.get_key_for_metadata()
            raise CheckVibrationFailed(
                frequencies=freq[:1],
                label=f"Init={result_label}, Minima={k}",
                fqmin=fqmin,
            )

    return result, modes


def helper_optimization(
    config: Config,
    graph: Cluster | Gas | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_hash_change: bool = True,
    allow_not_connected: bool = False,
    raise_when_fail: bool = False,
    run_vibration: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[SysGraph | str, tuple[str | None, list[Atoms]], float]:
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
        the optimized graph,
        the tuple of (graph_id, optimization trajectory)
        and the time cost in seconds.
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
    last_pos: np.ndarray = lst[-1].get_positions()
    if np.any(np.isinf(last_pos)) or np.any(np.isnan(last_pos)):
        e = ValueError("Optimization failed, positions contain inf/nan.")
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, lst),
                perf_counter() - start,
            )
    if not coveraged:
        e = OptimizationFailed(
            type="minima",
            max_steps=max(int(config.optimizer.steps), len(lst)),
            label=graph_label,
        )
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, lst),
                perf_counter() - start,
            )

    # ---------------------------------------------
    #       check graph hash changed or not
    # ---------------------------------------------
    try:
        result = graph.update_geometry(
            new_positions=last_pos,
            parse_bonds=config.bonds,  # type: ignore
            deep=deep_copy,
        )
    except Exception as e:
        atomic_write(
            Path(f"debug-{graph_label}.xyz"),
            writer=lambda p: write_extxyz(p, lst),
        )
        raise e
    if (not allow_not_connected) and (not result.is_connected):
        e = HelperException(
            msg="graph is not connected after optimization",
            label=graph_label,
        )
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, lst),
                perf_counter() - start,
            )
    if not allow_hash_change and graph.hash != result.hash:
        e = HelperException(
            msg="hash changed after optimization",
            label=graph_label,
        )
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, lst),
                perf_counter() - start,
            )
    if not run_vibration:
        f: np.ndarray = lst[-1].get_forces(apply_constraint=True)
        result = result.update_energetics(
            energy=lst[-1].get_potential_energy(),
            fmax=np.linalg.norm(f, axis=1).max(),
            frequencies=None,
            deep=deep_copy,
        )
        return (
            result,
            (graph_label, lst),
            perf_counter() - start,
        )

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
        return (
            result,
            (graph_label, lst),
            perf_counter() - start,
        )
    except Exception as e:
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, lst),
                perf_counter() - start,
            )


def helper_dimer(
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_fixed_bonds_change: bool = False,
    allow_desorption: bool | Literal["all", "input"] = "all",
    allow_metal_element_in_gas: bool = False,
    displacement: np.ndarray | None = None,
    raise_when_fail: bool = False,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Reaction | Desorption | str, tuple[Any, list[Atoms]], float, str]:
    """Dimer search for transition state.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        allow_fixed_bonds_change: Whether to
            allow the fixed bonds change after dimer.
        allow_desorption: Whether to
            allow the desorption of the gas after dimer search.
                If 'all', then allow the desorption
                    of all the gas molecules.
                If 'input', then allow the desorption of
                    the gas molecules only in the input file.
        displacement: The displacement vector for dimer.
        raise_when_fail: Whether to raise exception when failed.
        deep_copy: Whether to deep copy the optimized graph.

    Raises:
        OptimizationFailed: If the dimer failed.
        CheckVibrationFailed: If the check of vibration failed.
        HelperException: If the fixed bonds changed after dimer search.

    Returns:
        the optimized graph,
        tuple of (graph_id and dimer trajectory),
        the time cost in seconds,
        and the result type.
    """
    result_type: str = "dimer"
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
    last_pos: np.ndarray = dimer_lst[-1].get_positions()
    if np.any(np.isinf(last_pos)) or np.any(np.isnan(last_pos)):
        e = ValueError("Optimization failed, positions contain inf/nan.")
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, dimer_lst),
                perf_counter() - start,
                result_type,
            )
    if not coveraged:
        e = OptimizationFailed(
            type="dimer",
            label=graph_label,
            max_steps=max(int(config.optimizer.steps * 1.5), len(dimer_lst)),
        )
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, dimer_lst),
                perf_counter() - start,
                result_type,
            )

    # -----------------------------------------
    #       check the fixed bond change or not
    # -----------------------------------------
    ts = graph.update_geometry(
        new_positions=last_pos,
        parse_bonds=config.bonds,  # type: ignore
        deep=deep_copy,
    )
    k = ts.get_key_for_metadata()
    if not allow_fixed_bonds_change:
        break_bonds, make_bonds = graph.bond_difference(ts)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, graph.idx_fix)):
            e = HelperException(
                msg="fixed bonds modified after dimer",
                label=f"{graph_label}, TS={k}",
            )
            if raise_when_fail:
                raise e
            else:
                return (
                    str(e),
                    (graph_label, dimer_lst),
                    perf_counter() - start,
                    result_type,
                )

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
    except Exception as e:
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, dimer_lst),
                perf_counter() - start,
                result_type,
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
        e = OptimizationFailed(
            type="product",
            label=graph_label,
            max_steps=max(int(config.optimizer.steps), len(opt_lst)),
        )
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, dimer_lst),
                perf_counter() - start,
                result_type,
            )
    conn_comp = product_result.connected_components
    assert len(conn_comp) >= 1, "Impossible !!!!"
    gas = None

    # ------------------------------------------------------------
    #       return/check if the product is not connected
    # ------------------------------------------------------------
    if len(conn_comp) >= 3:
        msg = f"The `product` is not connected ({len(conn_comp)} blocks)."
        e = HelperException(msg, label=graph_label)
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, dimer_lst),
                perf_counter() - start,
                result_type,
            )
    elif len(conn_comp) == 2:
        if not allow_desorption:
            msg = "but desorption event is not allowed."
            msg = f"The `product` is not connected ({msg})."
            e = HelperException(msg, label=graph_label)
            if raise_when_fail:
                raise e
            else:
                return (
                    str(e),
                    (graph_label, dimer_lst),
                    perf_counter() - start,
                    result_type,
                )
        else:
            gas_z = product_result.numbers[conn_comp[1]]
            gas_r = product_result.positions[conn_comp[1]]
            gas_structure = Structure(numbers=gas_z, positions=gas_r)
            if allow_metal_element_in_gas or gas_structure.is_nonmetal:
                pass  # go to next next block, ...
            else:
                k = gas_structure.symbols.get_chemical_formula("metal")
                msg = f"The second block of {k} has metal atoms."
                e = HelperException(msg, label=graph_label)
                if raise_when_fail:
                    raise e
                else:
                    return (
                        str(e),
                        (graph_label, dimer_lst),
                        perf_counter() - start,
                        result_type,
                    )

        # -----------------------------------------------------------
        # split gas object from disconnected product, and optimize it
        # -----------------------------------------------------------
        gas_atoms = Atoms(numbers=gas_z, positions=gas_r)  # type: ignore
        gas_init = Gas.from_ase(gas_atoms, parse_bonds=config.bonds)  # type: ignore
        gas_key, _gas_complete = gas_init.get_key_for_metadata(), False
        try:
            network = ReactionNetwork.from_config(
                config=config, remote_mode=True, restart=True
            )
            if network.contain_sysgraph(gas_key, "gas"):
                _gas_complete = True
                gas = network.read_sysgraph(gas_key, "gas")
            elif str(allow_desorption).lower() == "input":
                raise NotImplementedError("Not implemented yet.")
                gas_name_set = set(config.event.gas_sticking.keys())
                gas_name_set.update(config.initial_gas_pressure.keys())
                if gas_init.smiles not in gas_name_set:
                    msg = (
                        f"The gas({gas_init.smiles}) is not in the input file."
                    )
                    e = HelperException(msg, label=graph_label)
                    if raise_when_fail:
                        raise e
                    else:
                        return (
                            str(e),
                            (graph_label, dimer_lst),
                            perf_counter() - start,
                            result_type,
                        )
        except Exception:
            pass  # go to next next block, ...
        if not _gas_complete:
            try:
                gas, _, cost_time = helper_optimization(
                    config=config,
                    graph=gas_init,
                    allow_hash_change=False,
                    raise_when_fail=True,
                    deep_copy=deep_copy,
                    run_vibration=True,
                    **kwargs,
                )
                _gas_complete = True
            except Exception as e:
                msg = f"The gas({gas_init.smiles}) is not optimized. "
                msg += f"Fail reason: {str(e)}"
                e = HelperException(msg, label=graph_label)
                if raise_when_fail:
                    raise e
                else:
                    return (
                        str(e),
                        (graph_label, dimer_lst),
                        perf_counter() - start,
                        result_type,
                    )
        assert isinstance(gas, Gas), "Impossible !!!!"
        assert _gas_complete, "Impossible !!!!"

        # ------------------------------------------------------------
        # split biggest block from disconnected product as new product
        # ------------------------------------------------------------
        assert isinstance(product_result, Cluster), "Impossible !!!!"
        try:
            product, _, cost_time = helper_optimization(
                config=config,
                graph=Cluster.select(product_result, conn_comp[0]),
                allow_hash_change=True,
                raise_when_fail=True,
                deep_copy=deep_copy,
                run_vibration=False,
                **kwargs,
            )
            assert isinstance(product, Cluster), "Impossible !!!!"
        except Exception as e:
            key = product_result.get_key_for_metadata(use_positions_uuid=True)
            msg = f"The product({key}) is not optimized. Fail reason: {str(e)}"
            e = HelperException(msg, label=graph_label)
            if raise_when_fail:
                raise e
            else:
                return (
                    str(e),
                    (graph_label, dimer_lst),
                    perf_counter() - start,
                    result_type,
                )
        product_atoms = product.to_ase()
        product_result = product

    # ------------------------------------------------------------
    #       check the fixed bond change or not for product
    # ------------------------------------------------------------
    if product_result.hash == graph.hash:  # type: ignore
        key_r = graph.get_key_for_metadata(use_positions_uuid=True)
        key_p = product_result.get_key_for_metadata(use_positions_uuid=True)
        if key_r == key_p:
            e = HelperException(
                msg="product uuid is same as reactant",
                label=graph_label,
            )
            if raise_when_fail:
                raise e
            else:
                return (
                    str(e),
                    (graph_label, dimer_lst),
                    perf_counter() - start,
                    result_type,
                )
    if not allow_fixed_bonds_change:
        break_bonds, make_bonds = graph.bond_difference(product_result)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, product_result.idx_fix)):
            kp = product_result.get_key_for_metadata()
            e = HelperException(
                msg="fixed bonds modified for product",
                label=f"{graph_label}, P={kp}",
            )
            if raise_when_fail:
                raise e
            else:
                return (
                    str(e),
                    (graph_label, dimer_lst),
                    perf_counter() - start,
                    result_type,
                )

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
        if gas is None:
            rxn = Reaction(R=graph, P=product_result, T=ts)
        else:
            rxn = Desorption(R=graph, P=product_result, G=gas)
        return (
            rxn,
            (graph_label, dimer_lst),
            perf_counter() - start,
            result_type,
        )
    except Exception as e:
        if raise_when_fail:
            raise e
        else:
            return (
                str(e),
                (graph_label, dimer_lst),
                perf_counter() - start,
                result_type,
            )


def helper_adsorption(
    gas: Gas,
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    irun: int | None = None,
    initial_atoms: Atoms | None = None,
    allow_fixed_atoms_adsorption: bool = False,
    graph_label: Any | None = None,
    raise_when_fail: bool = False,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Adsorption | str, tuple[Any, list[Atoms]], float, str]:
    """Helper function for adsorption process.

    Returns:
        1. the adsorption object
        2. the graph label
        3. the cost time
        4. the result type
    """
    if graph_label is None:
        graph_label = graph.get_key_for_metadata()
        graph_label += f"_{gas.get_key_for_metadata()}"

    # -----------------------------------------
    #       call adsorption initial positions
    # -----------------------------------------
    assert graph.ncore > 0, "graph must have core"
    if initial_atoms is None:
        adsorption_helper = Helper(
            atoms=graph.to_ase(
                exclude_bond_attibutes=True,
                exclude_energetics=True,
            ).copy(),
            adsorbate=gas,
            core=np.unique(graph.idx_core),
            nfibonacci=int(config.exploration.nfibonacci),
            use_direct=True,
            use_raw=True,
        )
        if irun is None:
            irun = np.random.randint(0, adsorption_helper.nrun)
        initial_atoms = adsorption_helper.__call__(irun=irun)
    assert isinstance(initial_atoms, Atoms), "Impossible !"
    init_graph = graph.update_adsorption(
        Atoms(
            numbers=initial_atoms.numbers[graph.natoms :],
            positions=initial_atoms.positions[graph.natoms :],
        ),
        parse_bonds=config.bonds,  # type: ignore
        **kwargs,
    )

    # -----------------------------------------
    #       call optimization for adsorption
    # -----------------------------------------
    product_result, (_, lst), cost_time = helper_optimization(
        config=config,
        graph=init_graph,
        **(
            kwargs
            | dict(
                graph_label=graph_label,
                allow_hash_change=True,
                raise_when_fail=False,
                deep_copy=deep_copy,
                run_vibration=True,
            )
        ),
    )
    if isinstance(product_result, str):
        e = HelperException(msg=product_result, label=graph_label)
        if raise_when_fail:
            raise e
        else:
            return str(e), (graph_label, lst), cost_time, "adsorption"

    # ------------------------------------------------------------
    #       check the fixed bond change or not for product
    # ------------------------------------------------------------
    if not allow_fixed_atoms_adsorption:
        break_bonds, make_bonds = graph.bond_difference(product_result)
        diff_bonds = np.asarray(break_bonds + make_bonds)
        if np.any(np.isin(diff_bonds, product_result.idx_fix)):
            kp = product_result.get_key_for_metadata()
            e = HelperException(
                msg="fixed bonds modified for product",
                label=f"{graph_label}, P={kp}",
            )
            if raise_when_fail:
                raise e
            else:
                return (
                    str(e),
                    (graph_label, lst),
                    cost_time,
                    "adsorption",
                )

    rxn = Adsorption(R=graph, P=product_result, G=gas)
    return rxn, (graph_label, lst), cost_time, "adsorption"


def helper_match(
    config: Config,
    system: System,
    rxn_is_forward: bool,
    rxn_key: str,
) -> tuple[str, bool, np.ndarray | None]:
    """Helper function for matching process.

    Returns:
        1. the reaction key
        2. the reaction is forward or reversed
        3. the match result

    """
    network = ReactionNetwork.from_config(
        config=config, restart=True, remote_mode=True
    )
    info = network.read_event_info(rxn_key)
    if rxn_is_forward:
        result = system.get_match_mode(  # type: ignore
            pattern=network.read_sysgraph(info.key_r, "minima", True),
            algorithm="lad",
            return_match_target=True,
            only_number_color=False,
            only_count=False,
        )
    else:
        result = system.get_match_mode(  # type: ignore
            pattern=network.read_sysgraph(info.key_p, "minima", True),
            algorithm="lad",
            return_match_target=True,
            only_number_color=False,
            only_count=False,
        )
    assert not isinstance(result, int)
    return rxn_key, rxn_is_forward, result


def helper_apply(
    rxn_key: str,
    config: Config,
    system: System | Atoms,
    match_mode: np.ndarray,
    raise_when_fail: bool = True,
    forward: bool = True,
) -> tuple[str, bool, Atoms | str, float]:
    network = ReactionNetwork.from_config(
        config=config, restart=True, remote_mode=True
    )
    _, rxn = network.read_event(rxn_key)
    if not forward:
        rxn = rxn.reversed
    try:
        atoms, rmsd = rxn.apply(system, matched_indxs=match_mode)
        return rxn_key, forward, atoms, rmsd
    except Exception as e:
        if raise_when_fail:
            raise e
        else:
            return rxn_key, forward, str(e), np.inf
