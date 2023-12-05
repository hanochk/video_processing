import numpy as np
from nebula3_database.database.arangodb import DatabaseConnector
from nebula3_database.config import NEBULA_CONF
import cv2
from pathlib import Path
import csv
import requests

from nebula3_database.movie_db import MOVIE_DB
import tqdm
from PIL import Image

from nebula3_videoprocessing.videoprocessing.ontology_implementation import SingleOntologyImplementation
from blip import BLIP_Captioner
from nebula3_videoprocessing.videoprocessing.yolov7 import YoloTrackerModel
from nebula3_videoprocessing.videoprocessing.bboxes_implementation import DetectronBBInitter



class TokensPipeline:
    def __init__(self):
        self.config_db = NEBULA_CONF()
        self.db_host = self.config_db.get_database_host()
        self.database = self.config_db.get_playground_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.nre = MOVIE_DB()
        self.db = self.nre.db
        self.blip_captioner = BLIP_Captioner()
        self.ontology_objects = SingleOntologyImplementation('vg_objects', vlm_name="blip_itc")
        self.ontology_persons = SingleOntologyImplementation('persons', vlm_name="blip_itc")
        self.ontology_places = SingleOntologyImplementation('scenes', vlm_name="blip_itc")
        self.ontology_attributes = SingleOntologyImplementation('vg_attributes', vlm_name="blip_itc")
        self.yolo_detector = YoloTrackerModel()
        self.det_proposal = DetectronBBInitter()

    def load_img_url(self, img_url : str, pil_type=False):
        # Load PIL Image
        if pil_type:
            image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            if not image:
                raise Exception("Image couldn't be loaded successfully.")
        # Load OpenCV Image
        else:
            resp = requests.get(img_url, stream=True).raw
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    
    def compute_scores(self, ontology, img, top_n = 10):
        """
        Returns top n ontology list and its corresponding scores sorted in reverse order.
        """
        ontology_scores = ontology.compute_scores(img)
        sorted_scores = sorted(ontology_scores, key=lambda x: x[1], reverse=True)
        sorted_scores_ = [(score[0], str(score[1])) for score in sorted_scores]
        scores = sorted_scores_[:top_n]
        return scores

    def insert_json_to_db(self, combined_json, db_name="nebula_playground"):
        """
        Inserts a JSON with global & local tokens to the database.
        """

        self.nre.change_db("nebula_playground")
        self.db = self.nre.db

        query = 'UPSERT { movie_id: @movie_id } INSERT  \
                { movie_id: @movie_id, mdf: @mdf, roi: @roi, url: @url, global_objects: @global_objects, global_caption: @global_caption,\
                            global_persons: @global_persons, global_scenes: @global_scenes, source: @source\
                        } UPDATE {movie_id: @movie_id, mdf: @mdf, roi: @roi, url: @url, global_objects: @global_objects, global_caption: @global_caption,\
                            global_persons: @global_persons, global_scenes: @global_scenes, \
                            source: @source} IN s3_pipeline_tokens'

        self.db.aql.execute(query, bind_vars=combined_json)
        print("Successfully inserted to database.")
        return
        

    def create_json_global_tokens(self, movie_id, mdf, global_objects,
                                    global_caption, global_persons, global_scenes, img_url, source):
        """
        Returns a JSON filled with global tokens.
        """
        json_global_tokens = {
                    "movie_id": movie_id,
                    "mdf": mdf,
                    "global_objects": { 'blip' : global_objects },
                    "global_caption": { 'blip' : global_caption },
                    "global_persons": { 'blip' : global_persons },
                    "global_scenes":  { 'blip' : global_scenes  },
                    "url": img_url,
                    "source": source
        }
        return json_global_tokens

    def create_json_local_tokens(self, movie_id, mdf, local_dict, img_url, source="None"):
        
        json_local_tokens = {
            "movie_id": movie_id,
            "mdf": mdf,
            "roi": local_dict,
            "url": img_url,
            "source": source
        }
        return json_local_tokens

    def create_combined_json(self, glob_tkns_json, loc_tkns_json):
        """
        Returns a JSON filled with global and local tokens.
        """
        combined_json = glob_tkns_json | loc_tkns_json
        return combined_json
    

    def create_local_tokens(self, img_url):
        """
        Returns a JSON with local tokens for an image url.
        """
        cv_img = self.load_img_url(img_url, pil_type=False)
        yolo_bboxes = self.yolo_detector.forward(cv_img)

        cv_img = self.load_img_url(img_url, pil_type=False)
        bbox_proposals = self.det_proposal.compute_bbox_proposals(cv_img)
        bb_rescale_ratio = [inp/out for out,inp in zip(bbox_proposals['image_size'][:2], cv_img.shape)]
        pil_img = self.load_img_url(img_url, pil_type=True)

        bbox_propsals_objs = []
        bbox_propsals_attrs = []

        local_dict = []

        for idx, bbox in enumerate(bbox_proposals['meta_data_det']):
            scaled_bbox = [bbox[0]*bb_rescale_ratio[1], bbox[1]*bb_rescale_ratio[0],
                            bbox[2]*bb_rescale_ratio[1], bbox[3]*bb_rescale_ratio[0]]

            # Objects on ROI proposals.
            scores_obj = self.ontology_objects.compute_scores_with_bboxes(pil_img, scaled_bbox)  
            scores_obj_sorted = sorted(scores_obj, key=lambda x: x[1], reverse=True)
            scores_obj_sorted_ = [(score[0], str(score[1])) for score in scores_obj_sorted]

            bbox_propsals_objs.append({str(scaled_bbox) : scores_obj_sorted_})

            # Attributes on ROI proposals.
            scores_obj = self.ontology_attributes.compute_scores_with_bboxes(pil_img, scaled_bbox)  
            scores_attr_sorted = sorted(scores_obj, key=lambda x: x[1], reverse=True)
            scores_attr_sorted_ = [(score[0], str(score[1])) for score in scores_attr_sorted]
            
            bbox_propsals_attrs.append({str(scaled_bbox) : scores_attr_sorted_})

            img = self.load_img_url(img_url, pil_type=True)
            cropped_image = img.crop((scaled_bbox[0], scaled_bbox[1], scaled_bbox[2], scaled_bbox[3]))
            processed_frame = self.blip_captioner.process_frame(cropped_image, convert=False)
            bbox_caption = self.blip_captioner.generate_caption(processed_frame)

            local_dict.append({
                'roi_id': str(idx),
                'bbox': str(bbox),
                'bbox_source': 'rpn',
                'local_captions': {'blip': bbox_caption},
                'local_objects': {'blip' : scores_obj_sorted_},
                'local_attributes': {'blip':scores_attr_sorted_},
            })


        
        
        json_local_tokens = self.create_json_local_tokens(movie_id = "123456", mdf="1", local_dict=local_dict,
                                                            img_url=img_url, source="None")
                                                            

        return json_local_tokens






    def create_global_tokens(self, img_url):
        """
        Returns a JSON with global tokens for an image url.
        """
        img = self.load_img_url(img_url, pil_type=True)

        scores_objects = self.compute_scores(self.ontology_objects, img, top_n = 10)
        scores_persons = self.compute_scores(self.ontology_persons, img, top_n = 10)
        scores_places = self.compute_scores(self.ontology_places, img, top_n = 10)

        img = self.load_img_url(img_url, pil_type=True)
        processed_frame = self.blip_captioner.process_frame(img, convert=False)
        caption = self.blip_captioner.generate_caption(processed_frame)

        json_global_tokens = self.create_json_global_tokens(movie_id = "123456", mdf="1", global_objects=scores_objects,
                                                        global_caption=caption, global_persons=scores_persons,
                                                         global_scenes=scores_places, img_url=img_url, source="None")

        return json_global_tokens


    









def main():
    tokens_pipeline = TokensPipeline()
    img_url = 'https://cs.stanford.edu/people/rak248/VG_100K/2316634.jpg'
    glob_tkns_json = tokens_pipeline.create_global_tokens(img_url)
    loc_tkns_json = tokens_pipeline.create_local_tokens(img_url)
    combined_json = tokens_pipeline.create_combined_json(glob_tkns_json, loc_tkns_json)
    tokens_pipeline.insert_json_to_db(combined_json)
    a=0



if __name__ == '__main__':
    main()