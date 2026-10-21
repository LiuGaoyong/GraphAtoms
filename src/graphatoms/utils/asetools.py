from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Any

import ase.optimize
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.io.trajectory import TrajectoryWriter
from ase.mep import DimerControl, MinModeAtoms, idpp_interpolate
from ase.mep import interpolate as interpolate_images
from ase.mep.dimer import DimerTranslate, MinModeTranslate
from ase.mep.neb import NEB, BaseNEB, DyNEB
from ase.optimize.optimize import Dynamics, Optimizer
from ase.units import invcm
from ase.vibrations import Vibrations, VibrationsData

_ERR_MSG = "Optimization failed because of infinite position."
OPTIMIZE_METHODS: dict[str, type[Optimizer]] = {}
for k in ase.optimize.__all__:
    v = getattr(ase.optimize, k)
    if isinstance(v, type) and issubclass(v, Optimizer):
        OPTIMIZE_METHODS[k] = v


def call_vib(
    atoms: Atoms,
    calc: Calculator,
    *,
    ignore_fqmin: float = 10.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Vibration atoms (frequencies in cm^-1 & harmonic modes).

    Parameters
    ----------
    atoms : Atoms
        The atoms to analyze.
    calc : Calculator
        The calculator to use.
    ignore_fqmin : float, optional
        The minimum frequency to ignore. Defaults to 10.0.
        Effect: the element frequency will be set to 1e-5 if the
          absolute value of element frequency is smaller than this value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The frequencies in cm^-1 and the harmonic modes.
    """
    if calc is not None:
        atoms.calc = calc
    assert atoms.calc is not None, "Please set calculator."
    atoms.calc.reset()

    with TemporaryDirectory() as tmpdir:
        vib = Vibrations(atoms, name=tmpdir)
        # print(f"Vib(id): {vib.indices}")
        vib.run()
        vibdata: VibrationsData = vib.get_vibrations()
    eng, modes = vibdata.get_energies_and_modes(all_atoms=True)
    freq = np.asarray(eng / invcm, dtype=complex)
    freq = np.real(freq) - np.imag(freq)  # complex to real
    freq[np.abs(freq) < abs(ignore_fqmin)] = 1e-5
    return freq, modes


def call_optimization(
    atoms: Atoms,
    calc: Calculator,
    *,
    method: str = "LBFGS",
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    max_steps: int = 100,
    fmax: float = 0.01,
) -> tuple[list[Atoms], bool]:
    """Optimize the `ase.Atoms`.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    method : str, optional
        The method to use. Defaults to "LBFGS".
    max_steps : int, optional
        The maximum number of steps. Defaults to 100.
    fmax : float, optional
        The maximum force. Defaults to 1e-4.

    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.
    """
    assert method in OPTIMIZE_METHODS, (
        f"Method {method} not found in `ase.optimize`."
    )
    OPT_CLS: type[Optimizer] = OPTIMIZE_METHODS[method]
    result_lst: list[Atoms] = []
    converged = False

    # NOTE: `atoms` can also be an `ase.mep.neb.BaseNEB` (see `call_neb`).
    # It is optimized like an `Atoms`, but it only provides the
    # `get_positions()` method instead of the `positions` attribute.
    atoms.calc = calc
    atoms.calc.reset()
    optimizer: Optimizer = OPT_CLS(
        atoms,
        trajectory=trajectory,
        append_trajectory=append_trajectory,
        logfile=logfile,
    )
    optimizer.attach(
        _trajectory_record,
        container=result_lst,
        object=optimizer,
        interval=1,
    )
    try:
        for _ in range(max_steps):
            converged = optimizer.run(steps=1, fmax=fmax)
            positions: np.ndarray = atoms.get_positions()  # type: ignore
            if np.any(np.isinf(positions)):
                raise RuntimeError(_ERR_MSG)
            if np.any(np.isnan(positions)):
                raise RuntimeError(_ERR_MSG)
            if converged:
                break
    except RuntimeError as e:
        converged = False
        raise e
    return result_lst, converged


def call_dimer_displace(
    atoms: Atoms,
    calc: Calculator,
    *,
    parse_mask_from_atoms: bool = True,
    mask: list[bool] | np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Call dimer method to search transition state."""
    lst, _ = call_dimer(
        atoms=atoms,
        calc=calc,
        mask=mask,
        parse_mask_from_atoms=parse_mask_from_atoms,
        max_steps=2,
    )
    return lst[1].positions - lst[0].positions


def call_dimer(
    atoms: Atoms,
    calc: Calculator,
    *,
    displacement: np.ndarray | None = None,
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    parse_mask_from_atoms: bool = True,
    mask: list[bool] | np.ndarray | None = None,
    max_steps: int = 1000,
    fmax: float = 0.02,
    **kwargs,
) -> tuple[list[Atoms], bool]:
    """Call dimer method to search transition state.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    displacement : np.ndarray | None, optional
        The displacement vector. Defaults to None.
    mask : list[bool] | np.ndarray | None, optional
        Which atoms will be moved during displacement. Defaults to None.
    parse_mask_from_atoms : bool, optional
        Whether to parse mask from atoms. Defaults to True.
    max_steps : int, optional
        The maximum number of steps. Defaults to 1000.
    fmax : float, optional
        The maximum force. Defaults to 0.01.
    debug : bool, optional
        Whether to print debug information. Defaults to False.
        If True, the optimization will print log information to stdout.

    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.

    """
    if mask is None and parse_mask_from_atoms:
        fixed_indices = []
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fixed_indices.extend(constr.get_indices())
        fixed_indices = list(set(fixed_indices))
        mask = [i not in fixed_indices for i in range(len(atoms))]
    else:
        mask = np.asarray(mask, dtype=bool).tolist()
    if calc is not None:
        atoms.calc = calc
    assert atoms.calc is not None, "Please set calculator."

    param: dict[str, Any] = kwargs.copy()
    param.update({"logfile": logfile, "mask": mask})
    if displacement is not None:
        param["initial_eigenmode_method"] = "displacement"
        param["displacement_method"] = "vector"
        displacement = np.asarray(displacement, copy=True)

    # record initial energy
    atoms.calc.reset()
    atoms.get_potential_energy()
    target = atoms.copy()
    target.calc = SPC(target, **atoms.calc.results)
    target.info["dimer_curvature"] = np.inf
    data: list[Atoms] = [target]
    if trajectory is not None:
        mode = "a" if append_trajectory else "w"
        traj = TrajectoryWriter(trajectory, mode)
        traj.write(atoms=data[-1])
    else:
        traj = None
    converged = False

    # run dimer
    with DimerControl(**(param | kwargs)) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)
        d_atoms.displace(displacement_vector=displacement)
        with MinModeTranslate(d_atoms, logfile=param["logfile"]) as dim_rlx:
            for _ in range(max_steps):
                _trajectory_record(container=data, object=dim_rlx)
                if traj is not None:
                    traj.write(atoms=data[-1])
                converged = dim_rlx.run(fmax=fmax, steps=1)
                if np.any(np.isinf(data[-1].positions)):
                    raise RuntimeError(_ERR_MSG)
                if np.any(np.isnan(atoms.positions)):
                    raise RuntimeError(_ERR_MSG)
                if converged:
                    _trajectory_record(container=data, object=dim_rlx)
                    if traj is not None:
                        traj.write(atoms=data[-1])
                        traj.close()
                    break
    return data, converged


def build_neb_images(
    atoms: Atoms,
    calc: Calculator,
    final_atoms: Atoms,
    intermediate_atoms: Atoms | None = None,
    *,
    nimages: int = 5,
    method4interpolate: str = "linear",
    method4opt: str = "FIRE",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> list[Atoms]:
    """Build the images of the NEB band.

    The `atoms` (i.e. the initial state) and the `final_atoms` (i.e. the
    final state) are optimized first.  If `intermediate_atoms` is given, it
    is used as an (unrelaxed) waypoint in the middle of the band, so that
    the band goes `initial -> intermediate -> final`; otherwise the interior
    images are interpolated directly between the two optimized states.

    Parameters
    ----------
    atoms : Atoms
        The initial state.
    calc : Calculator
        The calculator to use for relaxing the endpoints.
    final_atoms : Atoms
        The final state.
    intermediate_atoms : Atoms | None, optional
        The intermediate state, i.e. the waypoint of the band.
        Defaults to None.
    nimages : int, optional
        The number of images of the band (must be >= 3). Defaults to 5.
    method4interpolate : str, optional
        The method to interpolate the interior images, "linear" or "idpp".
        Only "linear" is supported when `intermediate_atoms` is given,
        because the band then crosses the waypoint.
    method4opt : str, optional
        The optimizer for relaxing the endpoints. Defaults to "FIRE".
    max_steps : int, optional
        The maximum number of steps for relaxing the endpoints.
    fmax : float, optional
        The maximum force for relaxing the endpoints.

    Returns
    -------
    list[Atoms]
        The images of the NEB band, from the initial to the final state.
    """
    assert nimages >= 3, "nimages must be at least 3."
    images: list[Atoms] = []
    for k, at in zip(["initial", "final"], [atoms, final_atoms]):
        lst, converged = call_optimization(
            at,
            calc,
            method=method4opt,
            max_steps=max_steps,
            trajectory=None,
            logfile=None,
            fmax=fmax,
        )
        if not converged:
            raise RuntimeError(f"Optimization did not converge for {k} image.")
        images.append(lst[-1])
    assert len(images) == 2
    first, last = images[0], images[-1]

    # the first image is the optimized initial state, the last image is the
    # optimized final state, and the interior images are interpolated.
    result: list[Atoms] = [first.copy() for _ in range(nimages - 1)]
    result.append(last)
    if intermediate_atoms is None:
        interpolate_images(result)
        if method4interpolate == "idpp":
            idpp_interpolate(result, traj=None, log=None)
        elif method4interpolate != "linear":
            raise NotImplementedError(
                f"Unknown method4interpolate: {method4interpolate}"
            )
    else:
        if method4interpolate != "linear":
            raise NotImplementedError(
                "Only `method4interpolate='linear'` is supported "
                + "when `intermediate_atoms` is given."
            )
        # piecewise linear interpolation through the intermediate state
        pos_init, pos_final = first.positions, last.positions
        pos_middle = np.asarray(intermediate_atoms.positions, dtype=float)
        imiddle: int = nimages // 2
        for k in range(1, nimages - 1):
            if k < imiddle:
                t = k / imiddle
                pos = pos_init * (1.0 - t) + pos_middle * t
            elif k == imiddle:
                pos = pos_middle
            else:
                t = (k - imiddle) / (nimages - 1 - imiddle)
                pos = pos_middle * (1.0 - t) + pos_final * t
            result[k].set_positions(pos)
    return result


def run_neb(
    images: list[Atoms],
    calc: Calculator,
    *,
    climb: bool = True,
    k4spring: float = 0.1,
    dynamic_relaxation: bool = True,
    method4spring: str = "improvedtangent",
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    method4opt: str = "FIRE",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> tuple[list[Atoms], bool]:
    """Run the NEB optimization on the given images.

    Parameters
    ----------
    images : list[Atoms]
        The images of the NEB band (see `build_neb_images`).
    calc : Calculator
        The calculator to use.
    climb : bool, optional
        Whether to climb. Defaults to True.
    k4spring : float, optional
        The spring constant. Defaults to 0.1.
    dynamic_relaxation : bool, optional
        Whether to use dynamic relaxation. Defaults to True.
        True skips images with forces below the convergence criterion.
        This is updated after each force call; if a previously converged
        image goes out of tolerance (due to spring adjustments between
        the image and its neighbors), it will be optimized again.
        False reverts to the default NEB implementation.
    method4spring : str, optional
        The method of the spring forces. Defaults to "improvedtangent".
    method4opt : str, optional
        The optimizer. Defaults to "FIRE".
    max_steps : int, optional
        The maximum number of steps. Defaults to 100.
    fmax : float, optional
        The maximum force. Defaults to 0.05.

    Returns
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.
    """
    assert len(images) >= 3, "The NEB band must have at least 3 images."
    for at in images:
        at.calc = deepcopy(calc)
        at.calc.reset()
        at.get_potential_energy()
    neb = DyNEB(
        images,
        method=method4spring,
        dynamic_relaxation=dynamic_relaxation,
        climb=climb,
        k=k4spring,
        # Note: the `fmax` of `DyNEB` must be identical to the one of the
        # optimizer, otherwise the dynamic relaxation skips the wrong images.
        fmax=fmax,
    )
    return call_optimization(
        neb,  # type: ignore
        calc,
        method=method4opt,
        max_steps=max_steps,
        trajectory=trajectory,
        append_trajectory=append_trajectory,
        logfile=logfile,
        fmax=fmax,
    )


def call_neb(
    atoms: Atoms,
    calc: Calculator,
    final_atoms: Atoms,
    intermediate_atoms: Atoms | None = None,
    *,
    nimages: int = 5,
    climb: bool = True,
    k4spring: float = 0.1,
    dynamic_relaxation: bool = True,
    method4spring: str = "improvedtangent",
    method4interpolate: str = "linear",
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    method4opt: str = "FIRE",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> tuple[list[Atoms], bool]:
    """Call NEB method to search transition state.

    The band is built by `build_neb_images` (the given `intermediate_atoms`
    is used as the waypoint of the band if it is not None) and optimized by
    `run_neb`.

    Parameters
    ----------
    atoms : Atoms
        The initial state.
    calc : Calculator
        The calculator to use.
    final_atoms : Atoms
        The final state.
    intermediate_atoms : Atoms | None, optional
        The intermediate state (the waypoint of the band). Defaults to None.
    nimages : int, optional
        The number of images. Defaults to 5. Must be >= 3.
    climb : bool, optional
        Whether to climb. Defaults to True.
    k4spring : float, optional
        The spring constant. Defaults to 0.1.
    dynamic_relaxation : bool, optional
        Whether to use dynamic relaxation. Defaults to True.
        True skips images with forces below the convergence criterion.
        This is updated after each force call; if a previously converged
        image goes out of tolerance (due to spring adjustments between
        the image and its neighbors), it will be optimized again.
        False reverts to the default NEB implementation.
    method4spring : str | None, optional
        The method to use for spring optimization. Defaults to None.
        Choice betweeen five methods:

            * aseneb: legacy ase NEB implementation
            * improvedtangent: Paper I NEB implementation (default)
            * eb: Paper III full spring force implementation
            * spline: Paper IV spline interpolation (supports precon)
            * string: Paper IV string method (supports precon)

    method4interpolate : str, optional
        The method to interpolate the interior images. Defaults to "linear".
        'linear' provides a standard straight-line interpolation, while
        'idpp' uses an image-dependent pair potential.  Only 'linear' is
        supported when `intermediate_atoms` is given.

    Paper I:
        G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
        :doi:`10.1063/1.1323224`

    Paper II:
        G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
        113, 9901 (2000).
        :doi:`10.1063/1.1329672`

    Paper III:
        E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
        145, 094107 (2016)
        :doi:`10.1063/1.4961868`

    Paper IV:
        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465


    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.
    """
    images: list[Atoms] = build_neb_images(
        atoms=atoms,
        calc=calc,
        final_atoms=final_atoms,
        intermediate_atoms=intermediate_atoms,
        nimages=nimages,
        method4interpolate=method4interpolate,
        method4opt=method4opt,
        max_steps=max_steps,
        fmax=fmax,
    )
    return run_neb(
        images=images,
        calc=calc,
        climb=climb,
        k4spring=k4spring,
        dynamic_relaxation=dynamic_relaxation,
        method4spring=method4spring,
        logfile=logfile,
        trajectory=trajectory,
        append_trajectory=append_trajectory,
        method4opt=method4opt,
        max_steps=max_steps,
        fmax=fmax,
    )


def _trajectory_record(container: list[Atoms], object: Dynamics) -> None:
    """Record the trajectory of the dynamics object.

    Parameters
    ----------
    container : list[Atoms]
        The container to store the trajectory.
    object : Dynamics
        The dynamics object to record.

    Returns: None
    -------------
    """
    if isinstance(object, (DimerTranslate, MinModeTranslate)):
        d_atoms = object.atoms
        assert isinstance(d_atoms, MinModeAtoms)
        atoms0: Atoms = d_atoms.get_atoms()
        assert isinstance(atoms0.calc, Calculator)
        d_atoms.get_potential_energy()
        target = atoms0.copy()
        target.calc = SPC(target, **atoms0.calc.results)
        target.info["dimer_curvature"] = d_atoms.get_curvature()
        container.append(target)
    elif isinstance(object, Dynamics):
        if isinstance(object.atoms, (NEB, DyNEB, BaseNEB)):
            for atoms in object.atoms.images:
                target = atoms.copy()
                assert isinstance(atoms.calc, Calculator)
                target.calc = SPC(target, **atoms.calc.results)
                container.append(target)
        else:
            target = object.atoms.copy()
            assert isinstance(object.atoms.calc, Calculator)
            target.calc = SPC(target, **object.atoms.calc.results)
            container.append(target)
    else:
        raise ValueError(f"Unknown type {type(object)}.")
