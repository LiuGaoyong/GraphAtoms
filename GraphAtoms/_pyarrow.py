from typing import Annotated, Any, cast, get_args

import pydantic_to_pyarrow.schema as pdt2pa
from pyarrow import Schema
from pydantic import BaseModel


def get_pyarrow_schema(
    pydantic_class: type[BaseModel],
    allow_losing_tz: bool = False,
    exclude_fields: bool = False,
    by_alias: bool = False,
) -> Schema:
    """Converts a Pydantic model into a PyArrow schema.

    Args:
        pydantic_class (Type[BaseModelType]): The Pydantic model class to convert.
        allow_losing_tz (bool, optional): Whether to allow losing timezone information
            when converting datetime fields. Defaults to False.
        exclude_fields (bool, optional): If True, will exclude fields in the pydantic
            model that have `Field(exclude=True)`. Defaults to False.
        by_alias (bool, optional): If True, will create the pyarrow schema using the
            (serialization) alias in the pydantic model. Defaults to False.

    Returns:
        pa.Schema: The PyArrow schema representing the Pydantic model.
    """
    settings = pdt2pa.Settings(
        allow_losing_tz=allow_losing_tz,
        by_alias=by_alias,
        exclude_fields=exclude_fields,
    )
    for name, field_info in pydantic_class.model_fields.items():
        if field_info.exclude and settings.exclude_fields:
            continue
        field_type = field_info.annotation
        metadata = field_info.metadata
        if field_type is None:
            # Not sure how to get here through pydantic, hence nocover
            raise pdt2pa.SchemaCreationError(
                f"Missing type for field {name}"
            )  # pragma: no cover

        nullable = False
        if pdt2pa._is_optional(field_type):
            nullable = True
            types_under_union = list(set(get_args(field_type)) - {type(None)})
            # mypy infers field_type as Type[Any] | None here, hence casting
            field_type = cast(type[Any], types_under_union[0])
        print(nullable, field_type)
        print(nullable, metadata)

        # try:
        #     pa_field = _get_pyarrow_type(field_type, metadata, settings)
        # except Exception as err:  # noqa: BLE001 - ignore blind exception
        #     raise SchemaCreationError(
        #         f"Error processing field {name}: {field_type}, {err}"
        #     ) from err

    #     serialized_name = name
    #     if settings.by_alias and field_info.serialization_alias is not None:
    #         serialized_name = field_info.serialization_alias
    #     fields.append(pa.field(serialized_name, pa_field, nullable=nullable))

    # return pa.schema(fields)


if __name__ == "__main__":
    from GraphAtoms.utils.ndarray import NDArray, numpy_validator

    class Atoms(BaseModel):
        numbers: Annotated[NDArray, numpy_validator("uint8")]
        positions: (
            Annotated[NDArray, numpy_validator(float, (-1, 3))] | None
        ) = None

    print(get_pyarrow_schema(Atoms))
