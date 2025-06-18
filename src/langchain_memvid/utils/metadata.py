from typing import Iterable, TypeVar, Type

T = TypeVar("T")


def get_on_first_match(metadata: dict, *keys: Iterable[str], default: T = None, expected_type: Type[T] = None) -> T:
    """Get the first value from the metadata dictionary that is not None and optionally matches the expected type."""
    # Derive type from default if expected_type not provided and default is not None
    if expected_type is None and default is not None:
        expected_type = type(default)

    for key in keys:
        if key in metadata and (value := metadata[key]) is not None:
            if expected_type is None or isinstance(value, expected_type):
                return value

    return default
