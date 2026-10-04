import pytest

from ragbasics.registry import available, build, register


def test_register_and_build_passes_params():
    @register("test_widget", "plain")
    class Widget:
        def __init__(self, size: int = 1):
            self.size = size

    assert build("test_widget", "plain", size=3).size == 3
    assert available("test_widget") == ["plain"]


def test_duplicate_name_is_rejected():
    register("test_dupe", "x")(dict)
    with pytest.raises(ValueError, match="already registered"):
        register("test_dupe", "x")(dict)


def test_unknown_name_lists_what_is_available():
    register("test_gadget", "known")(dict)
    with pytest.raises(KeyError, match="known"):
        build("test_gadget", "unknown")
