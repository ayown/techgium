# risk_engines/base.py

from abc import ABC, abstractmethod


class BaseRiskEngine(ABC):
    """
    Abstract base class for all physiological risk engines.

    Every engine MUST:
    - have a system name
    - implement run()
    - return a standardized result dictionary
    """

    def __init__(self, system_name: str):
        self.system = system_name

    @abstractmethod
    def run(self) -> dict:
        """
        Run the risk evaluation and return a standardized result.
        """
        pass
        