from tkinter import N
from typing import NewType
from abc import ABC, abstractmethod
from PIL import Image

Bbox = NewType('Bbox', list) # @TODO: add list length 4

ScoredBbox = NewType('ScoredBbox', tuple[Bbox, float])

class VgInterface(ABC):

    def __init__(self):
        super().__init__() 

    @abstractmethod
    def ground_objects(self, image : Image, text : str) -> list[ScoredBbox]:
        pass

    @abstractmethod
    def ground_objects_batch(self, image : Image, texts : list[str]) -> list[list[ScoredBbox]]:
        pass