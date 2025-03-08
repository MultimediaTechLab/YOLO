import omegaconf
from typing import List


def build_config(overrides: List[str] = []) -> omegaconf.DictConfig:
    """
    Creates an explicit config for testing.

    Example:
        >>> from yolo.utils.config_utils import build_config
        >>> cfg = build_config(overrides=['task=train'])
        >>> cfg = build_config(overrides=['task=validation'])
        >>> cfg = build_config(overrides=['task=inference'])
    """
    import yolo
    import os
    import pathlib
    from hydra import compose, initialize

    # This is annoying that we cant just specify an absolute path when it is
    # robustly built. Furthermore, the relative path seems like it isn't even
    # from the cwd, but the module that is currently being run.

    # Find the path that we need to be relative to in a somewhat portable
    # manner (i.e. will work in a Jupyter snippet).
    try:
        path_base = pathlib.Path(__file__).parent
    except NameError:
        path_base = pathlib.Path.cwd()
    yolo_path = pathlib.Path(yolo.__file__).parent
    rel_yolo_path = pathlib.Path(os.path.relpath(yolo_path, path_base))
    # rel_yolo_path = yolo_path.relative_to(path_base, walk_up=True)  # requires Python 3.12
    config_path = os.fspath(rel_yolo_path / 'config')
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg
