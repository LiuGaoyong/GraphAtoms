from time import perf_counter
from typing import override

from ase.calculators.calculator import Calculator

from graphatoms.reaction import Desorption, Reaction
from graphatoms.system import Cluster, Gas  # type: ignore
from graphatoms.system.database import DatabaseABC
from graphatoms.utils import asetools
from graphatoms.utils.parser import hydra_parse

from ._0abc import BaseABC
from ._helper import helper_dimer


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
        futures: list = []

        # -----------------------------------------
        # submit dimer tasks to executor
        # -----------------------------------------
        for _ in range(int(self.config.exploration.maxtry)):
            thetacutoff = float(self.config.exploration.thetacutoff)
            if thetacutoff < 0:
                futures.append(
                    self.executor.submit(
                        helper_dimer,
                        config=self.config,
                        graph_label=cluster_key,
                        allow_fixed_bonds_change=False,
                        graph=cluster.model_copy(deep=True),
                        raise_when_fail=False,
                        displacement=None,
                    )
                )
            else:
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
                        self.executor.submit(
                            helper_dimer,
                            config=self.config,
                            graph_label=cluster_key,
                            allow_fixed_bonds_change=False,
                            graph=cluster.model_copy(deep=True),
                            raise_when_fail=False,
                            displacement=disp,
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
        newold = self.network.recorder.exploration[cluster_key]
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            event, _, cost_time = future_result
            if isinstance(event, str):
                newold.fail += 1
                msg: str = "DimerSearch(failed) "
            elif isinstance(event, Reaction | Desorption):
                msg: str = "DimerSearch(success) "
                if self.network.write(event):  # event is new
                    newold.continuous_old = 0
                    newold.new += 1
                else:
                    newold.continuous_old += 1
                    newold.old += 1
            else:
                raise ValueError(f"Unknown event type: {type(event)}")
            m = f"N={newold.new},O={newold.old},F={newold.fail}"
            m = f"({m},S={newold.skip},C={newold.continuous_old})"
            self.logger.info(f"{msg}{cost_time:.2f} for {event}. {m}")
            confidence = self.config.exploration.maxconfidence
            if newold.exploration_can_be_finished(confidence):
                break
        for future in futures:
            future.cancel()

        self.network.persistence()


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
