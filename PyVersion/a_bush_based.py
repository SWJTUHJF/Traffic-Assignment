from __future__ import annotations

from .alg_base_solver import BaseSolver


class BushBased(BaseSolver):
    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError
