"""Configure pytest warning filters."""

import warnings


def pytest_configure(config):
    """Configure pytest - add specific warning filters."""
    # Ignore numpy deprecation warning about core._multiarray_umath
    warnings.filterwarnings(
        "ignore",
        message="numpy.core._multiarray_umath is deprecated",
        category=DeprecationWarning,
        module="faiss.loader"
    )

    # Ignore SWIG-related warnings about missing __module__ attributes
    for type_name in ["SwigPyPacked", "SwigPyObject", "swigvarlink"]:
        warnings.filterwarnings(
            "ignore",
            message=f"builtin type {type_name} has no __module__ attribute",
            category=DeprecationWarning,
            module="importlib._bootstrap"
        )
