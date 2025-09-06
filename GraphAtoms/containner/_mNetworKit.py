from abc import ABC, abstractmethod

import networkit as nk
from typing_extensions import Self


class GraphMixinNetworKit(ABC):
    @abstractmethod
    def as_networkit(self) -> nk.Graph: ...
    @classmethod
    @abstractmethod
    def from_networkit(cls, graph: nk.Graph) -> Self: ...
