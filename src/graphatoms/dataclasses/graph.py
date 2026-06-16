from pydantic import BaseModel

from ._pydanticMixin import PydanticConvertFactoryMixin


class Graph(BaseModel, PydanticConvertFactoryMixin):
    pass
