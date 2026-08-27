"""Packaged default configuration for Certus."""

from importlib import resources


def default_policy_path() -> str:
    """Return a filesystem path to the packaged default policy YAML.

    Useful as a starting point: ``shutil.copy(default_policy_path(), "policy.yaml")``.
    """
    with resources.as_file(resources.files(__package__) / "default_policy.yaml") as path:
        return str(path)


__all__ = ["default_policy_path"]
