from dataclasses import dataclass
from .constants import OUTPUT_JSON, TYPE_MOVIE


# @dataclass
# class MovieEntry:
#     movie_id: str
#     url_path: str
#     orig_url: str
#             "scenes": scenes, # [[0,375]],
#             "scene_elements": scene_elements,
#             "mdfs": mdfs,
#             "meta": movie_meta,
#             "updates": 1,
#             "source": "external"
#     id: str # "22094333" / "Movies/308719/1/0"
#     type: str # 'image" / "frame"
#     expert: str = None # "places"
#     bbox: list = None
#     label: str = None
#     label_meta: dict = None
#     score: float = None
#     re_id: int = 0
