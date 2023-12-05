import typing
from abc import ABC, abstractmethod
from PIL import Image
class VlmInterface(ABC):

    def __init__(self):
        super().__init__() 

    @abstractmethod
    def compute_similarity_url(self, url : str, text : list[str]) -> list[float]:
        pass
    
    @abstractmethod
    def compute_similarity(self, image : Image, text : list[str]) -> list[float]:
        pass
