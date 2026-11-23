import biomedqa


def test_package_imports_and_has_version() -> None:
    assert isinstance(biomedqa.__version__, str)
    assert biomedqa.__version__
