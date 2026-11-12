from scripts.smoke_test import main


def test_smoke_script_module_is_importable():
    assert callable(main)
