from array_api_compat import array_namespace
from array_api_compat.common._typing import Namespace

from .typing import Array, ArrayNamespace


def get_namespace(
    *xs: Array | complex | None,
    use_compat: bool | None = None,
) -> ArrayNamespace | Namespace:
    return array_namespace(
        *xs,  # type: ignore[reportOptionalMemberAccess]
        use_compat=use_compat,
        # api_version="2024.12",
    )  # type: ignore[reportOptionalMemberAccess]
