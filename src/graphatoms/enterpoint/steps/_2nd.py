from concurrent.futures import Future
from time import perf_counter
from typing import Any, override

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.parallel import get_executor, wait_one
from graphatoms.reaction import Desorption, Reaction
from graphatoms.system import Cluster, Gas  # type: ignore
from graphatoms.system.database import DatabaseABC
from graphatoms.utils import asetools
from graphatoms.utils.parser import hydra_parse

from ._0abc import BaseABC


class SecondStepSurface(BaseABC):
    """The class for exploring the surface process."""

    @override
    def run(self, cluster: Cluster) -> None:
        cluster_key = f"Cluster({DatabaseABC.get_key_of(cluster)})"
        assert cluster.check_minima(
            fmax=float(self.config.event.max_force),
            fqmin=float(self.config.event.min_frequency),
        ), f"Cluster {cluster_key} is not at a minimum."
        calc: Calculator = hydra_parse(
            self.config.calculator,  # type: ignore
            Calculator,
        )

        start: float = perf_counter()
        with get_executor(self.pmode, max_workers=self.pworkers) as executor:
            futures: list[Future[tuple[Any, float]]] = []

            # -----------------------------------------
            # submit dimer tasks to executor
            # -----------------------------------------
            for _ in range(int(self.config.exploration.maxtry)):
                disp = asetools.call_dimer_displace(
                    atoms=cluster.to_ase().copy(),
                    calc=calc,
                    mask=None,
                    parse_mask_from_atoms=True,
                    start=start,
                )
                can_be_skip, cosine = self.network.scheduler.can_be_skip(
                    cluster_key,
                    diffpositions=disp,
                    thetacutoff=float(self.config.exploration.thetacutoff),
                )
                if can_be_skip:
                    self.network.recorder.exploration[cluster_key].skip += 1
                    msg = "Skip to submit dimer task for "
                else:
                    msg = "Submit dimer task for "
                    futures.append(
                        executor.submit(
                            self.helper_dimer,
                            config=self.config,
                            cluster=cluster,
                            displacement=disp,
                            raise_on_failed=False,
                        )
                    )

                self.logger.info(f"{msg}{cluster_key}, cosine={cosine:.2f}")
            self.logger.info(
                f"Submit {len(futures)} dimer tasks by "
                f"{perf_counter() - start:.2f} seconds"
            )

            # -----------------------------------------
            # wait for the dimer tasks to finish
            # -----------------------------------------
            while len(futures) > 0:
                future_result, futures = wait_one(futures)
                event, cost_time = future_result
                if isinstance(event, str):
                    self.network.recorder.exploration[cluster_key].fail += 1
                    msg: str = "DimerSearch(failed),CostTime="
                elif isinstance(event, Reaction | Desorption):
                    msg: str = "DimerSearch(success),CostTime="
                    if self.network.write(event):  # event is new
                        self.network.recorder.exploration[cluster_key].new += 1
                    else:
                        self.network.recorder.exploration[cluster_key].old += 1
                else:
                    raise ValueError(f"Unknown event type: {type(event)}")
                self.logger.info(f"{msg}{cost_time:.2f} for {event}.")

                confidence = self.config.exploration.maxconfidence
                newold = self.network.recorder.exploration[cluster_key]
                if newold.exploration_can_be_finished(confidence):
                    break
            for future in futures:
                future.cancel()

        self.network.persistence()

    @staticmethod
    def helper_dimer(
        config: Config,
        cluster: Cluster,
        *args,
        displacement: np.ndarray | None = None,
        raise_on_failed: bool = False,
        **kwargs,
    ) -> tuple[Reaction | Desorption | str, float]:
        """Helper function for exploring the dimer process."""
        cluster_key = f"Cluster({DatabaseABC.get_key_of(cluster)})"
        assert cluster.check_minima(
            fmax=float(config.event.max_force),
            fqmin=float(config.event.min_frequency),
        ), f"Cluster {cluster_key} is not at a minimum."
        calc: Calculator = hydra_parse(
            config.calculator,  # type: ignore
            Calculator,
        )
        start: float = perf_counter()

        # -----------------------------------------
        # call dimer for TS search
        # -----------------------------------------
        dimer_lst, coveraged = asetools.call_dimer(
            atoms=cluster.to_ase().copy(),
            calc=calc,
            displacement=displacement,
            logfile=None,
            trajectory=None,
            append_trajectory=False,
            parse_mask_from_atoms=True,
            mask=None,
            max_steps=int(config.optimizer.steps),
            fmax=float(config.event.max_force),
            **kwargs,
        )
        if not coveraged:
            msg = f"Optimization dimer (failed): {cluster_key}."
            if raise_on_failed:
                raise BaseABC.OptimizationFailed(msg)
            else:
                return msg, perf_counter() - start

        # -----------------------------------------
        # call vibration for dimer result
        # -----------------------------------------
        freqs, vib_modes = asetools.call_vib(atoms=dimer_lst[-1], calc=calc)
        float(config.event.min_frequency_for_ts)
        f = dimer_lst[-1].get_forces()
        ts = Cluster.from_ase(
            dimer_lst[-1],
            parse_bonds=config.bonds,  # type: ignore
            parse_bonds_distance=False,
            parse_bonds_order=False,
            energy=dimer_lst[-1].get_potential_energy(),
            fmax=np.linalg.norm(f, axis=1).max(),
            frequencies=freqs,
            nadsorbate=0,
        )
        if not ts.check_ts(
            fmax=float(config.event.max_force),
            fqmin=float(config.event.min_frequency_for_ts),
        ):
            msg = f"Vibration (failed) for TS: {cluster_key}."
            freqs_str = ",".join(f"{f:.2f}" for f in freqs[:3])
            msg += f"freqs=({freqs_str},...)"
            if raise_on_failed:
                raise BaseABC.CheckVibrationFailed(msg)
            else:
                return msg, perf_counter() - start

        # -----------------------------------------
        # descend the minimum frequency mode
        # -----------------------------------------
        vdiff = dimer_lst[-1].positions - cluster.positions
        ldiff = np.linalg.norm(vdiff, axis=1)
        ldiff[ldiff < 1e-5] = np.inf
        imin_ldiff = np.argmin(ldiff)
        lmin_mode = np.linalg.norm(vib_modes[0][imin_ldiff])
        mode = vib_modes[0] * vdiff[imin_ldiff] / lmin_mode
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
            )
            if coveraged:
                opt_result = Cluster.from_ase(
                    opt_lst[-1],
                    parse_bonds=config.bonds,  # type: ignore
                    parse_bonds_distance=False,
                    parse_bonds_order=False,
                    energy=None,
                    fmax=None,
                    frequencies=None,
                    nadsorbate=0,
                )
                if opt_result.hash != cluster.hash:  # type: ignore
                    product_atoms = opt_lst[-1]
                    break
        if product_atoms is None:
            msg = f"Optimization dimer for product (failed): {cluster_key}."
            if raise_on_failed:
                raise BaseABC.OptimizationFailed(msg)
            else:
                return msg, perf_counter() - start

        # -----------------------------------------
        # call vibration for product
        # -----------------------------------------
        freqs, _ = asetools.call_vib(atoms=product_atoms, calc=calc)
        product = Cluster.from_ase(
            product_atoms,
            parse_bonds=config.bonds,  # type: ignore
            parse_bonds_distance=False,
            parse_bonds_order=False,
            energy=product_atoms.get_potential_energy(),
            fmax=np.linalg.norm(product_atoms.get_forces(), axis=1).max(),
            frequencies=freqs,
            nadsorbate=0,
        )
        if not product.check_minima(
            fmax=float(config.event.max_force),
            fqmin=float(config.event.min_frequency),
        ):
            msg = f"Vibration (failed) for product: {cluster_key}."
            freqs_str = ",".join(f"{f:.2f}" for f in freqs[:3])
            msg += f"freqs=({freqs_str},...)"
            if raise_on_failed:
                raise BaseABC.CheckVibrationFailed(msg)
            else:
                return msg, perf_counter() - start

        # ------------------------------------------------
        # check connected & construct Reaction/Desorption
        # ------------------------------------------------
        if not product.is_connected:
            # TODO: parse desorption event here
            msg = f"Product is not connected: {cluster_key}."
            if raise_on_failed:
                raise BaseABC.OptimizationFailed(msg)
            else:
                return msg, perf_counter() - start
        else:
            rxn = Reaction(R=cluster, P=product, T=ts)

        return rxn, perf_counter() - start


class SecondStepBulk(BaseABC):
    """The class for exploring the bulk process."""

    @override
    def run(self, cluster: Cluster) -> None:
        raise NotImplementedError


class SecondStepAdsorption(BaseABC):
    """The class for exploring the adsorption process."""

    @override
    def run(self, cluster: Cluster, gas: Gas) -> None:
        raise NotImplementedError
