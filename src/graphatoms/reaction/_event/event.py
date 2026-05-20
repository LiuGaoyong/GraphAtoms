from ._createMixin import CreateMiaxin
from ._ioMixin import IoMixin


class Event(CreateMiaxin, IoMixin):
    """The base class for all events in the reaction process.

    An event is a change of the system, which can be a reaction, a diffusion,
    or a desorption, etc. It is defined by the change of the system, which
    can be represented by the change of the graph.
    """

    pass
