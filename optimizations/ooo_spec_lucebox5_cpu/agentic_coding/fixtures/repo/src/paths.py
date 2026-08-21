from pathlib import PurePosixPath


def normalized_path(raw_path: str) -> str:
    """Return a repository-relative POSIX path without a leading './'."""
    return PurePosixPath(raw_path.removeprefix("./")).as_posix()
