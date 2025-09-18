from abc import ABC, abstractmethod

import networkx as nx
from typing_extensions import Self


class GraphMixinNetworkX(ABC):
    @abstractmethod
    def to_networkx(self) -> nx.Graph: ...
    @classmethod
    @abstractmethod
    def from_networkx(cls, graph: nx.Graph) -> Self: ...
