from pathlib import Path
from typing import Self, override

from ase import Atoms
from ase.io.trajectory import TrajectoryReader
from pydantic import model_validator

from ..base.rtgp import RTGP


class Event(RTGP):
    """The base class for all KMC events in the reaction process.

    An event is a change of the system, which can be a reaction, a diffusion,
    or a desorption, etc. It is defined by the change of the system, which
    can be represented by the change of the graph.
    """

    @override
    @classmethod
    def from_ase_trajectory(cls, traj: list[Atoms] | str | Path) -> Self:
        if not isinstance(traj, list):
            traj = list(TrajectoryReader(traj))  # type: ignore
        assert isinstance(traj, list), "The trajectory must be a list."
        if any(not isinstance(t, Atoms) for t in traj):
            raise ValueError("The trajectory must be a list of ase.Atoms.")
        raise NotImplementedError("Adsorption is not implemented.")

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert any(
            [
                self.is_reaction_LH,
                self.is_reaction_ER,
                self.is_adsorption,
                self.is_desorption,
            ]
        ), (
            "The event should be either a reaction based on "
            "Langmuir-Hinsher model, a reaction based on "
            "Eley-Rideal model, an adsorption or a desorption."
        )
        return self

    ########################################################################
    #           Properties for checking the type of the event.
    ########################################################################
    @property
    def is_reaction(self) -> bool:
        """Whether the event is a reaction."""
        n = int(max(len(self.R), len(self.P)))
        return self.T is not None and len(self.T) == n

    @property
    def is_reaction_LH(self) -> bool:
        """Whether the event is a reaction based on Langmuir-Hinsher model."""
        return self.is_reaction and self.G is None

    @property
    def is_reaction_ER(self) -> bool:
        """Whether the event is a reaction based on Eley-Rideal model."""
        return (
            self.is_reaction
            and self.G is not None
            and (
                len(self.P) == len(self.R) + len(self.G)  #
                or len(self.R) == len(self.P) + len(self.G)
            )
        )

    @property
    def is_adsorption(self) -> bool:
        """Whether the event is an adsorption."""
        return (
            self.T is None
            and self.G is not None
            and len(self.P) == len(self.R) + len(self.G)
        )

    @property
    def is_desorption(self) -> bool:
        """Whether the event is a desorption."""
        return (
            self.T is None
            and self.G is not None
            and len(self.R) == len(self.P) + len(self.G)
        )
