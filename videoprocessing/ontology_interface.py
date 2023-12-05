import typing
from abc import ABC, abstractmethod
class OntologyInterface(ABC):

    def __init__(self):
        super().__init__() 

    @abstractmethod
    def compute_scores(self, image) -> list[(str, float)]:
        pass
import typing
from abc import ABC, abstractmethod
class OntologyInterface(ABC):

    def __init__(self):
        super().__init__() 

    @abstractmethod
    def compute_scores(self, image) -> list[(str, float)]:
        pass
