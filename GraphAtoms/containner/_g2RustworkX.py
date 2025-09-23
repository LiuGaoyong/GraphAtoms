from abc import ABC, abstractmethod

import rustworkx as rx
from typing_extensions import Self


class GraphMixinRustworkX(ABC):
    @abstractmethod
    def to_rustworkx(self) -> rx.PyGraph: ...
    @classmethod
    @abstractmethod
    def from_rustworkx(cls, graph: rx.PyGraph) -> Self: ...
