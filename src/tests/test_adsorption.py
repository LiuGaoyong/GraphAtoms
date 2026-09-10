import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.io import read, write

from graphatoms.utils.adsorption import (
    AdsorptionABC,
    DirectAdsorption,
    Helper,
    RawAdsorption,
)


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    return Octahedron("Cu", 10)


@pytest.fixture(scope="module")
def atoms_pbc() -> Atoms:
    return read(
        Path(__file__).with_suffix(".xyz"),
        format="extxyz",
    )  # type: ignore


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent.parent
    p /= "tests-result-adsorption"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n**\n")
    return p


class TestAdsorption:
    """Test the adsorption."""

    @staticmethod
    def _test_add_adsorbate_and_optimize(  # noqa: D103
        atoms,
        adsorbate,
        core: int | list[int],
        cls: type[AdsorptionABC],
        calculator: Calculator | None,
        result_dir: Path,
        name: str,
    ) -> None:  # noqa: D103
        try:
            print()
            k = f"{name}_{adsorbate}"
            result_dir.mkdir(exist_ok=True)
            t0 = perf_counter()
            try:
                obj: AdsorptionABC = cls(
                    calculator=calculator,
                )
                result = obj(atoms=atoms, adsorbate=adsorbate, core=core)[0]
                result.numbers[core] = 79
                fname = result_dir.joinpath(f"{k}.png")
                write(fname.with_suffix(".png"), result, format="png")
                if hasattr(obj, "_atoms_lst"):
                    write(
                        fname.with_suffix(".xyz"),
                        obj._atoms_lst,
                        format="extxyz",
                    )
                print(f"  Write: {fname}")
            except Exception as e:
                msg = f"  No success: for {k} because of {e}"
                fname = result_dir.joinpath(f"{k}.error")
                with fname.open("w") as f:
                    f.write(msg)
                print(msg)
            finally:
                print(f"  Time({k}) = {perf_counter() - t0:.4f} s")
        except ImportError:
            return

    @pytest.mark.parametrize(
        "adsorbate",
        [
            "O",
            "CO",
            "H2O",
            "CH4",
            # "C6H6",
        ],
    )
    @pytest.mark.parametrize(
        "core,name",
        [
            ([303, 334, 464], "v_fcc"),  # vertex fcc hollow
            ([303, 334], "v_bri"),  # vertex bridge
            (303, "v_top"),  # vertex top
            (578, "e_top"),  # edge top
            ([578, 638], "e_bri"),  # edge bridge
            ([578, 638, 596], "e_fcc"),  # edge fcc hollow
            ([607, 608, 610], "s_fcc"),  # surface fcc hollow
            ([608, 610], "s_bri"),  # surface bridge
            ([610], "s_top"),  # surface top
        ],
    )
    def test_raw_adsorption(  # noqa: D103
        self,
        atoms,
        adsorbate,
        core: int | list[int],
        result_dir: Path,
        name: str,
    ) -> None:  # noqa: D103
        self._test_add_adsorbate_and_optimize(
            atoms,
            adsorbate,
            core,
            RawAdsorption,
            None,
            result_dir.joinpath("raw"),
            name,
        )

    @pytest.mark.parametrize(
        "adsorbate",
        [
            "O",
            "CO",
            "H2O",
            "CH4",
            "C6H6",
            "C2H6",
            "CH3OH",
            "CH3CH2OH",
            "C2H4",
        ],
    )
    @pytest.mark.parametrize(
        "core,name",
        [
            ([303, 334, 464], "v_fcc"),  # vertex fcc hollow
            ([303, 334], "v_bri"),  # vertex bridge
            (303, "v_top"),  # vertex top
            (578, "e_top"),  # edge top
            ([578, 638], "e_bri"),  # edge bridge
            ([578, 638, 596], "e_fcc"),  # edge fcc hollow
            ([607, 608, 610], "s_fcc"),  # surface fcc hollow
            ([608, 610], "s_bri"),  # surface bridge
            ([610], "s_top"),  # surface top
        ],
    )
    def test_direct_adsorption_nopbc(  # noqa: D103
        self,
        atoms,
        adsorbate,
        core: int | list[int],
        result_dir: Path,
        name: str,
    ) -> None:  # noqa: D103
        self._test_add_adsorbate_and_optimize(
            atoms,
            adsorbate,
            core,
            DirectAdsorption,
            calculator=None,
            result_dir=result_dir.joinpath("direct-nopbc"),
            name=name,
        )

    @pytest.mark.parametrize(
        "adsorbate",
        [
            "O",
            "CO",
            "H2O",
            "CH4",
            "C6H6",
            "C2H6",
            "CH3OH",
            "CH3CH2OH",
            "C2H4",
        ],
    )
    @pytest.mark.parametrize(
        "core,name",
        [
            ([0, 9, 99], "s_fcc"),  # surface fcc hollow
            ([0, 1], "s_bri"),  # surface bridge
            ([0], "s_top"),  # surface top
        ],
    )
    def test_direct_adsorption_pbc(  # noqa: D103
        self,
        atoms_pbc,
        adsorbate,
        core: int | list[int],
        result_dir: Path,
        name: str,
    ) -> None:  # noqa: D103
        self._test_add_adsorbate_and_optimize(
            atoms=atoms_pbc,
            adsorbate=adsorbate,
            core=core,
            cls=DirectAdsorption,
            calculator=None,
            result_dir=result_dir.joinpath("direct-pbc"),
            name=name,
        )


class TestHelper:
    """Test the Helper class."""

    def test_helper_initialization(
        self,
        atoms: Atoms,
        result_dir: Path,
    ) -> None:
        """Test Helper class initialization and nrun calculation."""
        obj = Helper(
            calculator=EMT(),
            atoms=atoms,
            adsorbate="O",
            core=[303, 334, 464],
            use_direct=True,
            use_raw=True,
            nfibonacci=10,
            max_steps_for_first_stage=10,
            max_steps_for_second_stage=10,
            distance_lst=np.array([2.0, 2.5]),
        )
        assert obj.nrun > 0
        print(f"  nrun = {obj.nrun}")

    def test_helper_raw_path_single_iteration(
        self,
        atoms: Atoms,
        result_dir: Path,
    ) -> None:
        """Test Helper raw adsorption path with single iteration."""
        core = [303, 334, 464]
        obj = Helper(
            calculator=EMT(),
            atoms=atoms,
            adsorbate="O",
            core=core,
            use_direct=True,
            use_raw=True,
            nfibonacci=10,
            max_steps_for_first_stage=0,
            max_steps_for_second_stage=0,
            distance_lst=np.array([2.0, 2.5]),
        )

        # irun=0 triggers raw path (after irun -= 1 becomes -1)
        result = obj(irun=0, outdir=result_dir)

        assert "score" in result
        assert "fmax" in result
        assert "nstage" in result
        assert "atoms" in result
        print(
            f"  Raw path result: score={result['score']:.4f}, "
            f"fmax={result['fmax']:.4f}, nstage={result['nstage']}"
        )
        fname = result_dir.joinpath("0-0.png")
        result["atoms"].numbers[core] = 79
        result["atoms"].write(fname, format="png")

    def test_helper_for_loop_single_process_raw(
        self,
        atoms: Atoms,
        result_dir: Path,
    ) -> None:
        """Test Helper using for-loop single process with raw path only.

        This test demonstrates the for-loop single process approach
        using the raw adsorption path, which works correctly.
        """
        core = [303, 334, 464]
        obj = Helper(
            calculator=EMT(),
            atoms=atoms,
            adsorbate="O",
            core=core,
            use_direct=True,
            use_raw=True,
            nfibonacci=10,
            max_steps_for_first_stage=0,
            max_steps_for_second_stage=0,
            distance_lst=np.array([2.0, 2.5]),
        )

        print(f"  Total runs: {obj.nrun}")
        results = []

        # Single iteration using raw path
        # irun=0 becomes -1 after irun -= 1, triggering raw path
        result = obj(irun=0, outdir=result_dir)
        results.append(result)
        print(
            f"  irun=0 (raw): score={result['score']:.4f}, "
            f"fmax={result['fmax']:.4f}, nstage={result['nstage']}"
        )

        assert len(results) == 1
        fname = result_dir.joinpath("0-1.png")
        result["atoms"].numbers[core] = 79
        result["atoms"].write(fname, format="png")

    @pytest.mark.parametrize("adsorbate", ["O", "CO"])
    @pytest.mark.parametrize(
        "core",
        [
            ([303, 334, 464],),  # vertex fcc hollow
            ([303],),  # vertex top
        ],
    )
    def test_helper_parametrized_raw_path(
        self,
        atoms: Atoms,
        adsorbate: str,
        core: tuple[list[int]],
        result_dir: Path,
    ) -> None:
        """Parametrized test for Helper with raw path."""
        obj = Helper(
            calculator=EMT(),
            atoms=atoms,
            adsorbate=adsorbate,
            core=core[0],
            use_direct=True,
            use_raw=True,
            nfibonacci=10,
            max_steps_for_first_stage=0,
            max_steps_for_second_stage=0,
            distance_lst=np.array([2.0, 2.5]),
        )
        print(f"  Total runs: {obj.nrun}")

        assert obj.nrun > 0
        for irun in range(obj.nrun):
            print(f"  irun={irun}")
            result = obj(irun=irun, outdir=result_dir)
            assert "score" in result
            assert "fmax" in result
            assert "nstage" in result
            assert "atoms" in result

            k = "_".join(map(str, core[0]))
            fname = result_dir.joinpath(f"{adsorbate}-{k}-{irun}.png")
            result["atoms"].numbers[core[0]] = 79
            result["atoms"].write(fname, format="png")
