"""The steps for the on-the-fly KMC simulation."""

from ase import Atoms

from ._1st import FirstStep
from ._2nd import SecondStepAdsorption as Adspt
from ._2nd import SecondStepBulk as Bulk
from ._2nd import SecondStepSurface as Surf
from ._3rd import ThirdStep


class OTFKMC(FirstStep, Adspt, Bulk, Surf, ThirdStep):
    """The class for the on-the-fly KMC simulation."""

    def run(self, steps: int = 1000000) -> None:  # type: ignore
        """Run the simulation for the given number of steps."""

        atoms: Atoms | None = None
        self.istep: int = 0

        for self.istep in range(steps):
            # 1 step: analyze the system
            dct = FirstStep.run(self, atoms)

            # 2 step: exploration
            ncore_4_adspt = int(self.config.exploration.max_ncore_for_surface)
            if bool(self.config.exploration.surface_only_explore_single_core):
                ncore_4_surface = 1
            else:
                ncore_4_surface = ncore_4_adspt
            for (is_surface, ncore, _), v in dct.items():
                if is_surface:
                    if ncore <= ncore_4_surface:
                        # exploration for surface cluster
                        Surf.run(self, cluster=v)
                    elif ncore <= ncore_4_adspt:
                        # exploration for adsorption process
                        for gas in self._gas_lst:
                            Adspt.run(self, cluster=v, gas=gas)
                else:
                    # exploration for bulk cluster
                    Bulk.run(self, cluster=v)

            # 3 step: update the system
            df = ThirdStep.run(self)
            print(df)
