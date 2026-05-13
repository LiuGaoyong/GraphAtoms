from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def hydra_parse(cfg: DictConfig, cls: type, debug: bool = False, **kw) -> Any:
    """Parse Hydra DictConfig to instantiate an object.

    Args:
        cfg: Hydra DictConfig object.
        cls: Target class type.
        debug: Whether to print config for debugging.
        **kw: Additional arguments for instantiate.

    Returns:
        Instance of cls.
    """
    if debug:
        print("=" * 64, "\nCONFIG:")
        print(OmegaConf.to_yaml(cfg))
    syscfg = OmegaConf.to_container(cfg, resolve=True)
    datamodule = instantiate(syscfg, **kw)
    assert isinstance(datamodule, cls)
    return datamodule
