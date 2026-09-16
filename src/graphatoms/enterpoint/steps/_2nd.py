from time import perf_counter
from typing import override

from ase.calculators.calculator import Calculator

from graphatoms.system import Cluster, Gas  # type: ignore
from graphatoms.utils import asetools
from graphatoms.utils.parser import hydra_parse

from ._0abc import BaseABC
from ._helper import helper_dimer


class SecondStepSurface(BaseABC):
    """The class for exploring the surface process."""

    @override
    def run(self, cluster: Cluster) -> None:
        cluster_key = cluster.get_key_for_metadata()
        assert cluster.check_minima(
            fmax=float(self.config.event.max_force),
            fqmin=float(self.config.event.min_frequency),
        ), f"Cluster {cluster_key} is not at a minimum."
        calc: Calculator = hydra_parse(
            self.config.calculator,  # type: ignore
            Calculator,
        )
        oldnew = self.network.recorder.cluster[cluster_key]
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
                    oldnew.found_skip()
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
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            event, _, cost_time = future_result
            label = self.network.found(
                event,
                for_cluster=cluster_key,
                for_system="",
                persist=True,
            )
            if label.startswith("fail"):
                self.logger.info(
                    f"Dimer search {label} by {cost_time:.2f}"
                    + f" seconds because of {event}"
                )
                oldnew.found_fail()
            else:
                msg = "Dimer search successfully, and got "
                msg += f"{label} by {cost_time:.2f} seconds."
                if str(event) not in label:
                    msg += f" Simplify original {event} by threshold "
                    msg += f"{self.config.event.simplified_threshold:.2f}"
                self.logger.info(msg)
                if "new" in label:
                    oldnew.found_new()
                else:
                    oldnew.found_old()
            confidence = self.config.exploration.maxconfidence
            if oldnew.exploration_can_be_finished(
                confidence=confidence,
                min_found=self.network.metadata.table.get_minconut_for(
                    cluster_key=cluster_key,
                ),
            ):
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
