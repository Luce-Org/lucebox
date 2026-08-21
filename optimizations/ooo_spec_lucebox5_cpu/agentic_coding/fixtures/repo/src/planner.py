from .paths import normalized_path


class ChangePlanner:
    def plan(self, candidate_paths: list[str]) -> list[str]:
        return sorted({normalized_path(path) for path in candidate_paths})
