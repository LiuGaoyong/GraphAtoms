# ruff: noqa: D100 D101 D102 D103


class NotSupportPeriodicSystem(KeyError):
    """The periodic system is not supported."""


class NotSupportNonOrthorhombicLattice(KeyError):
    """The non-orthorhombic lattice is not supported."""


def not_support_type(data, required_type: str) -> TypeError:
    return TypeError(
        f"Only support type({required_type:s}). But"
        f" the type of input is {type(data)} now."
    )
