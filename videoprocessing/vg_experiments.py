import sys
import os
import math
import random
import bisect
import pickle
import time
import numpy as np
from nebula3_database.database.arangodb import DatabaseConnector
from nebula3_database.config import NEBULA_CONF
# from movie_db import MOVIE_DB
import cv2
from pathlib import Path
import csv

from nebula3_database.movie_db import MOVIE_DB
import tqdm
from PIL import Image
from nebula3_videoprocessing.videoprocessing.ontology_implementation import SingleOntologyImplementation
from nebula3_videoprocessing.videoprocessing.vlm_factory import VlmFactory
import nebula_vg_driver.visual_genome.local as vg
import pandas as pd
import json
import itertools
import pandas as pd
from blip import BLIP_Captioner
import os.path

from nebula3_videoprocessing.videoprocessing.vlm_implementation import BlipItcVlmImplementation

# from nebula3_videoprocessing.videoprocessing.yolov7 import YoloTrackerModel
# from nebula3_videoprocessing.videoprocessing.detectron_tracker import DetectronModel

from nebula3_videoprocessing.videoprocessing.utils.config import config

import PIL.ImageDraw as ImageDraw

def open_json(json_path):
    with open(json_path) as f:
            data = json.load(f)
    return data

def open_csv(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

class VG_EXPERIMENT:
    def __init__(self):
        self.config_db = NEBULA_CONF()
        self.db_host = self.config_db.get_database_host()
        self.database = self.config_db.get_playground_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.nre = MOVIE_DB()
        self.nre.change_db("visualgenome")
        self.db = self.nre.db


    def vg_experiment_proposals(self, ontology_name = 'vg_objects', vlm_name='blip_itc'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_proposals_{ontology_name}_{vlm_name}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # idx = 0
        # for ob in sample_ids:
        #     image_id = images_data[sample_ids[idx]]['image_id']
        #     idx += 1

        from nebula3_videoprocessing.videoprocessing.bboxes_implementation import DetectronBBInitter
        ontology_imp = SingleOntologyImplementation(ontology_name, vlm_name)
        det_proposal = DetectronBBInitter()
        
        idx = 0
        # with open(os.path.join("/notebooks/vg_output", "results_proposals_vg_objects_blip_itc_vg.json"), "r") as f:
        #     results = json.load(f)

        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id", cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            if not os.path.exists(full_fname):
                continue
            cv_img = cv2.imread(full_fname)
            if cv_img is None:
                continue
            bbox_proposals = det_proposal.compute_bbox_proposals(cv_img)
            bb_rescale_ratio = [inp/out for out,inp in zip(bbox_proposals['image_size'][:2], cv_img.shape)]
            obj_dict = dict()
            obj_dict[cur_image_data['image_id']] = dict()
            for bbox in bbox_proposals['meta_data_det']:
                if not os.path.exists(full_fname):
                    break
                pil_img = Image.open(full_fname)
                if pil_img is None:
                    break
                scaled_bbox = [bbox[0]*bb_rescale_ratio[1], bbox[1]*bb_rescale_ratio[0],
                                bbox[2]*bb_rescale_ratio[1], bbox[3]*bb_rescale_ratio[0]]
                scores = ontology_imp.compute_scores_with_bboxes(pil_img, scaled_bbox)
                
                scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
                scores_sorted = [(score[0], str(score[1])) for score in scores_sorted]
                obj_dict[cur_image_data['image_id']].update( {str(scaled_bbox) : str(scores_sorted)} )
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)        
            idx += 1
    
    def vg_experiment_add_atrs_to_proposals(self, ontology_name='vg_attributes', vlm_name='blip_itc'):
        
        import ast
        det_proposals_path = "results_proposals_vg_objects_blip_itc_vg.json"

        det_proposals_path = os.path.join("/notebooks/vg_output", det_proposals_path)
        det_proposals = open_json(det_proposals_path)

        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_proposals_{ontology_name}_{vlm_name}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        ontology_imp = SingleOntologyImplementation(ontology_name, vlm_name)

        idx = 0
        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            cur_image_id = cur_image_data['image_id']
            print("Object id", cur_image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            if not os.path.exists(full_fname):
                continue
            cv_img = cv2.imread(full_fname)
            if cv_img is None:
                continue
            bbox_proposals = list(det_proposals[str(cur_image_id)].keys())
            obj_dict = dict()
            obj_dict[str(cur_image_data['image_id'])] = dict()
            for bbox in bbox_proposals:
                
                pil_img = Image.open(full_fname)
                if pil_img is None:
                    break
                bbox = ast.literal_eval(bbox)
                scores = ontology_imp.compute_scores_with_bboxes(pil_img, bbox)
                
                scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
                scores_sorted = [(score[0], str(score[1])) for score in scores_sorted]
                obj_dict[str(cur_image_data['image_id'])].update( {str(bbox) : str(scores_sorted)} )
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)        
            idx += 1


    def vg_experiment_caption(self, captioner='blip'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_paragraphs_{captioner}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)
        
        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        blip_captioner = BLIP_Captioner()
    
        idx = 0

        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id", cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            processed_frame = blip_captioner.process_frame(img)
            # print(f"URL: {cur_image_data['url']}")
            caption = blip_captioner.generate_caption(processed_frame)
            # print(f"caption: {caption}")
            obj_dict = dict()
            obj_dict.update({str(cur_image_data['image_id']) : caption})
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)        
            idx += 1


    def vg_experiment_topdown_yolov7(self, ontology_name = 'vg_objects'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_yolov7_nonunique_{ontology_name}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]

        
        # sample_ids = random.sample(range(1, len(image_ids) - 1), 1000)

        # with open(os.path.join(result_path, 'sample_ids.txt'), 'w') as f:
        #     for item in sample_ids:
        #         f.write("%s\n" % item)

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        # obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation(ontology_name, vlm_name)
        from nebula3_videoprocessing.videoprocessing.yolov7 import YoloTrackerModel
        yolo_detector = YoloTrackerModel()
        idx = 0

        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id", cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            # img = Image.open(full_fname)
            # scores = ontology_imp.compute_scores(img)
            img = cv2.imread(full_fname)
            if img is None:
                continue
            outputs = yolo_detector.forward(img)
            # unique_outputs = []
            # for output in reversed(outputs):
            #     cur_object= output.split(" ")[0]
            #     if not unique_outputs: # init
            #         unique_outputs.append(output)
            #     else:
            #         # Make sure the object isn't in the unique list, then add it
            #         obj_found = False
            #         for unique_output in unique_outputs:
            #             prev_object = unique_output.split(" ")[0]
            #             if cur_object == prev_object:
            #                 obj_found = True
            #         if not obj_found:
            #             unique_outputs.append(output)
            obj_dict = dict()
            obj_dict.update({str(cur_image_data['image_id']) : outputs})
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)        
            idx += 1
    

    def vg_experiment_topdown_detectron(self, ontology_name = 'vg_objects'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_detectron2_{ontology_name}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation(ontology_name, vlm_name)
        detectron2_detector = DetectronModel()
        idx = 0

        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id", cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            # img = Image.open(full_fname)
            # scores = ontology_imp.compute_scores(img)
            img = cv2.imread(full_fname)
            if img is None:
                continue
            outputs = detectron2_detector.forward(img, uniquify=True)
            unique_outputs = []
            for output in reversed(outputs):
                cur_object= output.split(" ")[0]
                if not unique_outputs: # init
                    unique_outputs.append(output)
                else:
                    # Make sure the object isn't in the unique list, then add it
                    obj_found = False
                    for unique_output in unique_outputs:
                        prev_object = unique_output.split(" ")[0]
                        if cur_object == prev_object:
                            obj_found = True
                    if not obj_found:
                        unique_outputs.append(output)
            obj_dict = dict()
            obj_dict.update({str(cur_image_data['image_id']) : unique_outputs})
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)        
            idx += 1
    
    def vg_experiment_ofa(self):

        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_ofa_vg_refcocog.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"

        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        # ipc_data = json.load(open('/storage/ipc_data/paragraphs_v1.json','r'))

        image_ids = [obj['image_id'] for obj in images_data]

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        from nebula3_experiments.vg_eval import Sg_handler
        from nebula3_vlmtokens_expert.vlmtokens.data.vg_dataset import bbox_xywh_to_xyxy
        vgenome_metadata = "/storage/vg_data/"
        sg_handler = Sg_handler(images_path=vgenome_metadata,  # data = json.load(open(image_data_dir + fname, 'r')) [(x['attribute']) for x in data['attributes']][1]['attributes']
                            image_data_dir=vgenome_metadata+'/by-id/',
                            synset_file=vgenome_metadata+'/synsets.json')
        idx = 0
        image_id_to_objs_bboxes = {}
        for sample_id in tqdm.tqdm(sample_ids):
            cur_image_data = str(images_data[sample_ids[idx]]['image_id'])
            sg = sg_handler.get_scene_graph(images_data[sample_ids[idx]]['image_id'])
            objs_and_bboxes_lst = []
            for i, (visual_objects, attrib) in enumerate(zip(sg.objects, sg.attributes)): #sg.objects[0].height
                h = visual_objects.height
                w = visual_objects.width
                y = visual_objects.y
                x = visual_objects.x
                xmin, ymin, xmax, ymax = bbox_xywh_to_xyxy((x,y,w,h))
                objs_and_bboxes_lst.append((visual_objects.names[0], [xmin, ymin, xmax, ymax]))
                # crop_image = image.crop((xmin, ymin, xmax, ymax))
                # width, height = crop_image.size
            image_id_to_objs_bboxes[cur_image_data] = objs_and_bboxes_lst
            idx += 1

        from nebula3_experts_vg.vg.vg_expert import VisualGroundingVlmImplementation
        vgnd = VisualGroundingVlmImplementation()
        results = dict()
        idx = 0
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id",  cur_image_data['image_id'])
            image_id = str(cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            img = img.convert('RGB')
            # print(full_fname)
            # sg = get_sc_graph(vg_ob['image_id'])
            obj_dict = dict()
            img_objects = []
            for obj_with_bboxes in image_id_to_objs_bboxes[image_id]:
                gt_obj = obj_with_bboxes[0]
                gt_bbox = obj_with_bboxes[1]
                bb, lprob = vgnd.compute_similarity(img, gt_obj)
                predicted_bbox = bb[0]['box']
                if image_id not in obj_dict :
                    obj_dict[image_id] = list()
                obj_dict[image_id].append([gt_obj, str(gt_bbox), str(float(lprob)), str(predicted_bbox)])
            results.update(obj_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout) 
            idx += 1  


    def vg_experiment_topdown(self, ontology_name = 'vg_objects', vlm_name='blip_itc'):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_{vlm_name}_{ontology_name}_vg.csv")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]

        
        # sample_ids = random.sample(range(1, len(image_ids) - 1), 1000)

        # with open(os.path.join(result_path, 'sample_ids.txt'), 'w') as f:
        #     for item in sample_ids:
        #         f.write("%s\n" % item)

        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        ontology_imp = SingleOntologyImplementation(ontology_name, vlm_name)

        idx = 0
        not_found_imgs = 0

        results = list()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            print("Object id", cur_image_data['image_id'])
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            if not os.path.isfile(full_fname):
                print("File not found {}".format(full_fname))
                not_found_imgs += 1
                print("Number of images that aren't found: {}".format(not_found_imgs))
                idx += 1
                continue
            img = Image.open(full_fname)
            scores = ontology_imp.compute_scores(img)
            obj_dict = dict()
            obj_dict.update({'image_id' : cur_image_data['image_id']})
            for i in range(len(scores)):
                obj_dict.update({ontology_imp.ontology[i]: scores[i][1]})
            results.append(obj_dict)
            # Intermediate save 
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(result_path, res_file), index=False)
            idx += 1
    
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(result_path, res_file), index=False)

    
    def vg_experiment_topdown_ensemble(self, ontology_name = 'vg_objects', vlm_names=['blip_itc', 'blip_itm']):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, f"results_{'_'.join(vlm_names)}_{ontology_name}_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]


        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        ontology_imp = MultipleOntologyImplementation(ontology_name, vlm_names)

        counter = 0
        results = dict()
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[counter]]
            image_id = cur_image_data['image_id']
            print("Object id", image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            scores = ontology_imp.compute_scores(img)
            sorted_scores = [sorted(vlm_scores, key=lambda x: x[1], reverse=True) for vlm_scores in scores]
            obj_dict = dict()
            obj_dict.update({image_id : {}})
            averaged_results = dict()
            averaged_results.update({image_id : {}})
            # Iterate over the VLMs Scores
            for idx, vlm_tuples in enumerate(sorted_scores):
                obj_dict[image_id].update({idx: {}})
                # if idx == 0:
                #     averaged_results[image_id].update({idx: {}})
                # Iterate over all the ontology and its scores from the VLM
                for i in range(0, len(vlm_tuples)):
                    obj_dict[image_id][idx].update({vlm_tuples[i][0]: i})
                    if idx == 0:
                        averaged_results[image_id].update({vlm_tuples[i][0]: i})
                    if idx > 0: # Make sure we check the results from at least the 2nd VLM outputs.
                        # Sum indices (Average later)
                        averaged_results[image_id][vlm_tuples[i][0]] = (obj_dict[image_id][idx][vlm_tuples[i][0]] +
                        obj_dict[image_id][idx - 1][vlm_tuples[i][0]])
            # Average results (We already summed, so just divide)
            for image_id in averaged_results:
                for key, val in averaged_results[image_id].items():
                # Make sure we don't divide by zero if all the results is first index
                    if averaged_results[image_id][key] == 0:
                        averaged_results[image_id][key] = 0
                    else:
                        averaged_results[image_id][key] = averaged_results[image_id][key] / len(sorted_scores)
            averaged_results = dict(sorted(averaged_results[image_id].items(), key=lambda item: item[1]), reverse=True)
            results.update({image_id : averaged_results})
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # Intermediate save 
            counter += 1

    
    def vg_experiment_topdown_paragraphs(self, vlm_name='blip_itc', with_split=False):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        if with_split:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg_split.json")
        else:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]


        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]

        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        vlm = VlmFactory().get_vlm(vlm_name)

        idx = 0
        results = {}
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            image_id = cur_image_data['image_id']
            print("Object id", image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            if with_split:
                paragraphs = cur_image_data['paragraph'].split('.')
                paragraphs = [paragraph for paragraph in paragraphs if paragraph]
            else:
                paragraphs = cur_image_data['paragraph']
            scores = vlm.compute_similarity(img, paragraphs)

            paragraphs_dict = dict()
            paragraphs_dict.update({image_id :{}})
            for i in range(len(scores)):
                paragraphs_dict[image_id].update({paragraphs[i]: str(scores[i])})
            results.update(paragraphs_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            idx += 1
    
    def vg_experiment_topdown_paragraphs_random(self, vlm_name='blip_itc', with_split=False):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        if with_split:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg_random_split.json")
        else:
            res_file = os.path.join(result_path, f"results_{vlm_name}_paragraphs_vg_random.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "results-visualgenome.json"), "r") as f:
            vg_objects = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids = [obj['image_id'] for obj in images_data]


        with open(os.path.join(result_path, "sample_ids.txt")) as f:
            sample_ids = [int(line.strip()) for line in f.readlines()]


        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        all_paragraphs = [img_data['paragraph'] for img_data in images_data]

        vlm = VlmFactory().get_vlm(vlm_name)

        idx = 0
        results = {}
        for vg_ob in tqdm.tqdm(sample_ids):
            cur_image_data = images_data[sample_ids[idx]]
            image_id = cur_image_data['image_id']
            print("Object id", image_id)
            fname = os.path.basename(cur_image_data['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            paragraphs = random.sample(all_paragraphs, 5)
            if with_split:
                paragraphs = [paragraph.split('.') for paragraph in paragraphs]
                paragraphs = list(itertools.chain.from_iterable(paragraphs))
                paragraphs = [paragraph.lstrip() for paragraph in paragraphs if len(paragraph) > 1]
            else:
                paragraphs = cur_image_data['paragraph']
           
            scores = vlm.compute_similarity(img, paragraphs)

            paragraphs_dict = dict()
            paragraphs_dict.update({image_id :{}})
            for i in range(len(scores)):
                paragraphs_dict[image_id].update({paragraphs[i]: str(scores[i])})
            results.update(paragraphs_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            idx += 1

    def clip_vg_relations_experiment(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_clip_vg_relations.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('clip')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})
            
        # idx = 0

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1
        
    def blip_vg_relations_experiment(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_blip_vg_relations.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('blip_itc')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})
            
        # idx = 0

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1
    
    def blip_vg_relations_experiment_random(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_blip_vg_relations_random.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_predicates.txt")) as f:
            vg_predicates = [line.strip() for line in f.readlines()]
        
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('blip_itc')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                random_predicate = random.choice(vg_predicates)
                triplet[1] = random_predicate
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1

    def clip_vg_relations_experiment_random(self):
        result_path = os.path.join(os.getcwd(), 'vg_output')
        res_file = os.path.join(result_path, "results_clip_vg_relations_random.json")
        vgenome_images = '/datasets/visualgenome/VG/'
        VG_DATA = "/notebooks/vg_data"
        with open(os.path.join(VG_DATA, "ipc_predicates.txt")) as f:
            vg_predicates = [line.strip() for line in f.readlines()]
        
        with open(os.path.join(VG_DATA, "ipc_triplet_relations.txt")) as f:
            vg_triplets = json.load(f)

        with open("/storage/ipc_data/paragraphs_v1.json", "r") as f:
            images_data = json.load(f)

        image_ids_to_index = [{obj['image_id']: idx} for idx, obj in enumerate(images_data) if str(obj['image_id']) in vg_triplets.keys()]
        # Load ontology
        obj_ontology_path = "/notebooks/nebula3_vlmtokens_expert/vlmtokens/visual_token_ontology/vg"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Created output directory")

        # ontology_imp = SingleOntologyImplementation('vg_objects', 'clip')
        vlm_imp = VlmFactory().get_vlm('clip')

        vg_triplets_parsed = {}
        for image_id, triplets in vg_triplets.items():
            str_triplets = []
            for triplet in triplets:
                random_predicate = random.choice(vg_predicates)
                triplet[1] = random_predicate
                str_triplet = ' '.join(triplet).lower()
                str_triplets.append(str_triplet)
            vg_triplets_parsed.update({image_id: str_triplets})

        results = {}
        for data in tqdm.tqdm(image_ids_to_index):
            image_id, cur_idx = list(data.items())[0][0], list(data.items())[0][1]
            print("Object id", image_id)
            fname = os.path.basename(images_data[cur_idx]['url'])
            full_fname = os.path.join(vgenome_images, fname)
            img = Image.open(full_fname)
            text = vg_triplets_parsed[str(image_id)]
            if text:
                scores = vlm_imp.compute_similarity(img, text)
            else:
                print(f"Skipped {image_id}")
                continue
            triplets_dict = dict()
            triplets_dict.update({image_id :{}})
            for i in range(len(scores)):
                triplets_dict[image_id].update({vg_triplets_parsed[str(image_id)][i]: str(scores[i])})
            results.update(triplets_dict)
            # Intermediate save 
            with open(res_file, 'w') as fout:
                json.dump(results, fout)
            # idx += 1
    
    def compute_average_relations(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        average_lst = []
        for image_id, triplet_to_scores in data.items():
            all_values = triplet_to_scores.values()
            all_values = [float(score) for score in all_values]
            avg = np.average(all_values) 
            average_lst.append(avg)
        output_avg = np.average(average_lst)
        print(f"Average: {output_avg}")

    def insert_vg_experiments_to_db(self, objects_path, captions_path, persons_path, scenes_path, ipc_path, db_name="nebula_playground", source='visualgenome'):
        self.nre.change_db(db_name)
        self.db = self.nre.db
        
        captions_data = open_json(captions_path)
        ipc_data = open_json(ipc_path)
        objects_data = open_csv(objects_path)
        persons_data = open_csv(persons_path)
        scenes_data = open_csv(scenes_path)

        img_ids_to_ontology = {}
        # Get all the image ids to the dict
        for img_id, _ in captions_data.items():
            img_ids_to_ontology.update({img_id: {"objects": {"blip": {}}, "captions": {"blip": {}}, "persons": {"blip": {}}, "scenes": {"blip": {}}}})

        idx = 1

        for img_id, _ in img_ids_to_ontology.items():
            
            # Get all the ontologies of the current image id
            captions = captions_data[img_id]
            # CSV to JSON
            persons_data_dict, scenes_data_dict, objects_data_dict = {}, {}, {}
            # Add all the ontology objects and their respective scores to the current image
            for j in range(1, len(persons_data[0])):
                if persons_data[idx][0] == img_id:
                    persons_data_dict.update({persons_data[0][j] : persons_data[idx][j]})
                else:
                    print("Error.")

             # Add all the ontology objects and their respective scores to the current image
            for j in range(1, len(scenes_data[0])):
                if scenes_data[idx][0] == img_id:
                    scenes_data_dict.update({scenes_data[0][j] : scenes_data[idx][j]})
                else:
                    print("Error.")
            
             # Add all the ontology objects and their respective scores to the current image
            for j in range(1, len(objects_data[0])):
                if objects_data[idx][0] == img_id:
                    objects_data_dict.update({objects_data[0][j] : objects_data[idx][j]})
                else:
                    print("Error.")
            idx +=1

            # a = sorted(objects_data_dict.items(), key=lambda x: x[1], reverse=True)
            # objects_data_dicts = [{key: val} for key,val in a]
            
            # objects_data_dicts = [{key,val} for key,val in objects_data_dict]
            # Insert all the ontologies to the current image id
            img_ids_to_ontology[img_id]['objects']['blip'] = [{"label": k, "score": float(v)} for (k, v) in sorted(objects_data_dict.items(), key=lambda x: x[1], reverse=True)]
            img_ids_to_ontology[img_id]['captions']['blip'] = captions
            img_ids_to_ontology[img_id]['persons']['blip'] = [{"label": k, "score": float(v)} for (k, v) in sorted(persons_data_dict.items(), key=lambda x: x[1], reverse=True)]
            img_ids_to_ontology[img_id]['scenes']['blip'] = [{"label": k, "score": float(v)} for (k, v) in sorted(scenes_data_dict.items(), key=lambda x: x[1], reverse=True)]


            query = 'UPSERT { image_id: @image_id } INSERT  \
                { image_id: @image_id, url: @url, global_objects: @global_objects, global_captions: @global_captions,\
                            global_persons: @global_persons, global_scenes: @global_scenes, source: @source\
                        } UPDATE {image_id: @image_id, url: @url, global_objects: @global_objects, global_captions: @global_captions,\
                            global_persons: @global_persons, global_scenes: @global_scenes, \
                            source: @source} IN s3_global_tokens1'
                            
            img_url = os.path.join('https://cs.stanford.edu/people/rak248/VG_100K', img_id + '.jpg')
            bind_vars = {
                            "image_id": int(img_id),
                            "global_objects": {"blip" : img_ids_to_ontology[img_id]['objects']['blip']},
                            "global_captions": {"blip" : img_ids_to_ontology[img_id]['captions']['blip']},
                            "global_persons": {"blip" : img_ids_to_ontology[img_id]['persons']['blip']},
                            "global_scenes": {"blip" : img_ids_to_ontology[img_id]['scenes']['blip']},
                            "url": img_url,
                            "source": source
                            }
            # print(bind_vars)
            print(idx)
            self.db.aql.execute(query, bind_vars=bind_vars)

    def insert_vg_local_experiments_to_db(self, objects_path, objects_blip_path, attributes_path, attributes_blip_path, caption_path, ipc_path, db_name="nebula_playground", source='visualgenome'):
            self.nre.change_db(db_name)
            self.db = self.nre.db
            
            ipc_data = open_json(ipc_path)
            objects_data = open_csv(objects_path)
            objects_blip_path = open_csv(objects_blip_path)
            attributes_data = open_csv(attributes_path)
            attributes_blip_path = open_csv(attributes_blip_path)
            caption_data = open_csv(caption_path)

            object_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in objects_data[1:]]

            object_blip_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in objects_blip_path[1:]]

            atrribute_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in attributes_data[1:]]

            atrribute_blip_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in attributes_blip_path[1:]]

            caption_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in caption_data[1:]]

            imgs_ids = {}

            img_ids_objs_to_ontology, img_ids_atts_to_ontology, img_ids_caps_to_ontology = {}, {}, {}
            img_ids_objs_blip_to_ontology, img_ids_atts_blip_to_ontology = {}, {}
            # Get all the image ids to the dict
            for img_id_dict in object_image_ids:
                img_id = list(img_id_dict.items())[0][0]
                img_ids_objs_to_ontology.update({img_id: {"objects": {"clip": {"rois": list()}}}})
                img_ids_atts_to_ontology.update({img_id: {"attributes": {"clip": {"rois": list()}}}})
                img_ids_caps_to_ontology.update({img_id: {"captions": {"blip": {"rois": list()}}}})
                img_ids_objs_blip_to_ontology.update({img_id: {"objects": {"blip": {"rois": list()}}}})
                img_ids_atts_blip_to_ontology.update({img_id: {"attributes": {"blip": {"rois": list()}}}})
                if img_id not in imgs_ids:
                    imgs_ids.update({img_id:''})


            for img_id_dict in object_blip_image_ids:
                img_id = list(img_id_dict.items())[0][0]
                objects_and_confidences = list(zip(objects_blip_path[0][:-2], img_id_dict[img_id][:-2]))
                objects_and_confidences = [(k, float(v)) for (k, v) in sorted(objects_and_confidences, key=lambda x: x[1], reverse=True)]
                roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
                img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois'].append({roi: objects_and_confidences})
                if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                    print(img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois'][0][roi][:5])

            # Get all the ROIs with their corresponding objects
            for img_id_dict in object_image_ids:
                img_id = list(img_id_dict.items())[0][0]
                objects_and_confidences = list(zip(objects_data[0][:-2], img_id_dict[img_id][:-2]))
                objects_and_confidences = [(k, float(v)) for (k, v) in sorted(objects_and_confidences, key=lambda x: x[1], reverse=True)]
                roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
                img_ids_objs_to_ontology[img_id]['objects']['clip']['rois'].append({roi: objects_and_confidences})
                if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                    print(img_ids_objs_to_ontology[img_id]['objects']['clip']['rois'][0][roi][:5])

            for img_id_dict in atrribute_blip_image_ids:
                img_id = list(img_id_dict.items())[0][0]
                attributes_and_confidences = list(zip(attributes_blip_path[0][:-2], img_id_dict[img_id][:-2]))
                attributes_and_confidences = [(k, float(v)) for (k, v) in sorted(attributes_and_confidences, key=lambda x: x[1], reverse=True)]
                roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
                img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois'].append({roi: attributes_and_confidences})
                if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                    print(img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois'][0][roi][:5])

            for img_id_dict in atrribute_image_ids:
                img_id = list(img_id_dict.items())[0][0]
                attributes_and_confidences = list(zip(attributes_data[0][:-2], img_id_dict[img_id][:-2]))
                attributes_and_confidences = [(k, float(v)) for (k, v) in sorted(attributes_and_confidences, key=lambda x: x[1], reverse=True)]
                roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
                img_ids_atts_to_ontology[img_id]['attributes']['clip']['rois'].append({roi: attributes_and_confidences})
                if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                    print(img_ids_atts_to_ontology[img_id]['attributes']['clip']['rois'][0][roi][:5])
            
            for img_id_dict in caption_image_ids:
                img_id = list(img_id_dict.items())[0][0]
                roi = img_id_dict[img_id][-2].replace("array([", "").replace("])","")
                caption = img_id_dict[img_id][0]
                img_ids_caps_to_ontology[img_id]['captions']['blip']['rois'].append({roi: caption})
    
            imgs_ids = list(imgs_ids.keys())
            # Iterate over all images ids
            # Insert all the ontologies to the current image id
            # img_ids_to_ontology[img_id]['objects']['blip'] = [{k: float(v)} for (k, v) in sorted(objects_data_dict.items(), key=lambda x: x[1], reverse=True)]
            # img_ids_to_ontology[img_id]['captions']['blip'] = captions
            # img_ids_to_ontology[img_id]['persons']['blip'] = [{k: float(v)} for (k, v) in sorted(persons_data_dict.items(), key=lambda x: x[1], reverse=True)]
            # img_ids_to_ontology[img_id]['scenes']['blip'] = [{k: float(v)} for (k, v) in sorted(scenes_data_dict.items(), key=lambda x: x[1], reverse=True)]
            # query = 'UPSERT { image_id: @image_id } INSERT  \
            #         { image_id: @image_id, url: @url, global_objects: @global_objects, global_captions: @global_captions,\
            #                     global_persons: @global_persons, global_scenes: @global_scenes, source: @source\
            #                 } UPDATE {image_id: @image_id, url: @url, global_objects: @global_objects, global_captions: @global_captions,\
            #                     global_persons: @global_persons, global_scenes: @global_scenes, \
            #                     source: @source} IN s3_local_tokens1'
            for idx, img_id in enumerate(imgs_ids):

                # Reset a fresh ROI dict every iteration
                # roi_dict = {
                #         'roi_id': 0,
                #         'bbox': '',
                #         'bbox_source': '',
                #         'local_captions': {'blip': []},
                #         'local_objects': {'clip': []},
                #         'local_attributes': {'clip': []}
                # }
                rois_list = []
                # Add objects & attributes ROIs
                objects_rois = img_ids_objs_to_ontology[img_id]['objects']['clip']['rois']
                attributes_rois = img_ids_atts_to_ontology[img_id]['attributes']['clip']['rois']
                captions_rois = img_ids_caps_to_ontology[img_id]['captions']['blip']['rois']
                attributes_blip_rois = img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois']
                objects_blip_rois = img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois']
                for ijx, obj_roi in enumerate(objects_rois):
                    roi_dict_temp = roi_dict = {
                        'roi_id': 0,
                        'bbox': '',
                        'bbox_source': '',
                        'local_captions': {'blip': []},
                        'local_objects': {'clip': [], 'blip': []},
                        'local_attributes': {'clip': [], 'blip': []}
                    }
                    roi = list(obj_roi.keys())[0]
                    roi_dict_temp['roi_id'] = ijx
                    roi_dict_temp['bbox'] = roi
                    roi_dict_temp['bbox_source'] = 'vg_gt'
                    roi_dict_temp['local_captions']['blip'] = captions_rois[ijx][roi]
                    roi_dict_temp['local_objects']['clip'] = objects_rois[ijx][roi]
                    roi_dict_temp['local_attributes']['clip'] = attributes_rois[ijx][roi]
                    roi_dict_temp['local_objects']['blip'] = objects_blip_rois[ijx][roi]
                    roi_dict_temp['local_attributes']['blip'] = attributes_blip_rois[ijx][roi] 
                    
                    rois_list.append(roi_dict_temp)



                query = 'UPSERT { image_id: @image_id } INSERT  \
                        { image_id: @image_id, roi: @roi, url: @url, source: @source \
                            } UPDATE {image_id: @image_id, roi: @roi, url: @url, source: @source} \
                                IN s3_local_tokens'

                img_url = os.path.join('https://cs.stanford.edu/people/rak248/VG_100K', img_id + '.jpg')
                bind_vars = {
                                "image_id": int(img_id),
                                "roi": rois_list,
                                "url": img_url,
                                "source": source
                                }
                # print(bind_vars)
                print(idx)
                self.db.aql.execute(query, bind_vars=bind_vars)
                a=0 
    

    def insert_local_yolo_tokens(self, yolo_path = "results_yolov7_vg_objects_vg.json", db_name="nebula_playground"):
            
            import re 
            self.nre.change_db(db_name)
            self.db = self.nre.db

            yolo_path = os.path.join("/notebooks/vg_output", yolo_path)
            yolo_data = open_json(yolo_path)

            source = 'yolov7'
            query = 'UPSERT { image_id: @image_id } INSERT  \
                    { image_id: @image_id, roi: @roi, url: @url, source: @source \
                        } UPDATE {image_id: @image_id, roi: @roi, url: @url, source: @source} \
                            IN s3_local_yolo'

            for img_id, objs_and_bboxes in yolo_data.items():
                
                temp_objs_and_bboxes = {}
                roi_list = []
                obj_conf_list = []
                for idx, obj_and_bbox in enumerate(objs_and_bboxes):

                    cur_bbox = re.findall("\[(.*?)\]", obj_and_bbox)[0].split(',')
                    cur_bbox = [float(bbox_val) for bbox_val in cur_bbox]
                    cur_obj =  obj_and_bbox.split(' ')[0]
                    cur_confidence = float(obj_and_bbox.split(' ')[-1])
                    temp_objs_and_bboxes = {
                                'roi_id': idx,
                                'bbox': cur_bbox,
                                'bbox_source': 'yolo',
                                'local_objects': {'yolo' : [(cur_obj, cur_confidence)]}
                            }
                    roi_list.append(temp_objs_and_bboxes)

                img_url = os.path.join('https://cs.stanford.edu/people/rak248/VG_100K', img_id + '.jpg')
                bind_vars = {
                                "image_id": int(img_id),
                                "roi": roi_list,
                                "url": img_url,
                                "source": source
                                }
                # print(bind_vars)
                self.db.aql.execute(query, bind_vars=bind_vars)
                a=0 

    def test_json(self):
        with open(("/notebooks/vg_output/results_yolov7_nonunique_vg_objects_vg.json"), "r") as f:
            data = json.load(f)
        obj_counter = 0
        for key, val in data.items():
            obj_counter += len(val)
        average = obj_counter / len(data)
        print("Average: ")
    
    def insert_local_proposals_tokens(self, det_proposals_path = "results_proposals_vg_objects_blip_itc_vg.json",
         det_proposals_attrs_path="results_proposals_vg_attributes_blip_itc_vg.json", db_name="nebula_playground"):
            
            import re 
            import ast
            self.nre.change_db(db_name)
            self.db = self.nre.db

            det_proposals_path = os.path.join("/notebooks/vg_output", det_proposals_path)
            det_proposals = open_json(det_proposals_path)

            det_proposals_attrs_path = os.path.join("/notebooks/vg_output", det_proposals_attrs_path)
            det_proposals_attrs = open_json(det_proposals_attrs_path)

            source = 'rpn'
            query = 'UPSERT { image_id: @image_id } INSERT  \
                    { image_id: @image_id, roi: @roi, url: @url, source: @source \
                        } UPDATE {image_id: @image_id, roi: @roi, url: @url, source: @source} \
                            IN s3_local_rpn1'
            counter = 0
            for img_id, objs_and_bboxes in det_proposals.items():
                temp_objs_and_bboxes = {}
                roi_list = []
                obj_conf_list = []
                for idx, obj_and_bbox in enumerate(objs_and_bboxes):

                    cur_bbox = re.findall("\[(.*?)\]", obj_and_bbox)[0].replace(" ","").split(",")
                    cur_bbox = [float(box) for box in cur_bbox]
                    objs_and_confidences = list(ast.literal_eval(objs_and_bboxes[obj_and_bbox]))
                    objs_and_confidences = [(obj_and_confidence[0], float(obj_and_confidence[1])) for obj_and_confidence in objs_and_confidences]
                    attrs_and_confidences = det_proposals_attrs[img_id][str(cur_bbox)]
                    attrs_and_confidences = list(ast.literal_eval(attrs_and_confidences))
                    attrs_and_confidences = [(att_and_confidence[0], float(att_and_confidence[1])) for att_and_confidence in attrs_and_confidences]
                    temp_objs_and_bboxes = {
                                'roi_id': idx,
                                'bbox': cur_bbox,
                                'bbox_source': 'rpn',
                                'local_objects': {'rpn' : objs_and_confidences},
                                'local_attributes': {'rpn': attrs_and_confidences}
                            }
                    roi_list.append(temp_objs_and_bboxes)

                img_url = os.path.join('https://cs.stanford.edu/people/rak248/VG_100K', img_id + '.jpg')
                bind_vars = {
                                "image_id": int(img_id),
                                "roi": roi_list,
                                "url": img_url,
                                "source": source
                                }
                # print(bind_vars)
                self.db.aql.execute(query, bind_vars=bind_vars)
                counter +=1
                print(counter)

    def test_json(self):
        with open(("/notebooks/vg_output/results_yolov7_nonunique_vg_objects_vg.json"), "r") as f:
            data = json.load(f)
        obj_counter = 0
        for key, val in data.items():
            obj_counter += len(val)
        average = obj_counter / len(data)
        print("Average: ")

    def process_ofa_results(self):

        # python -m spacy download en && python -m spacy download en_core_web_sm && python -m spacy download en_core_web_lg
        from nebula3_videoprocessing.videoprocessing.utils.image_utils import bb_intersection_over_union
        import ast
        import statistics
        from nebula3_experiments.vg_eval import VGEvaluation

        vg_eval = VGEvaluation()
        
        with open(("/notebooks/vg_output/results_ofa_vg_refcocog.json"), "r") as f:
            data = json.load(f)
        print(len(data))
        images_averages = []
        coco_classes = [
                        'person', 'bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
                        'fire hydrant', 'stop sign', 'parking meter','bench','bird','cat','dog','horse','sheep','cow',
                        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
                        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                        'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable',
                        'toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
                        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
                        ]
        idx=0
        for image_id, objs_and_bboxes in data.items():
            seen_objects_w_iou = {}
            for obj_and_bbox in objs_and_bboxes:
                gt_bbox = ast.literal_eval(obj_and_bbox[1])
                predicted_bbox = ast.literal_eval(obj_and_bbox[3] )
                cur_iou = bb_intersection_over_union(gt_bbox, predicted_bbox)
                current_obj = obj_and_bbox[0]
                # curent_obj = vg_eval.recall_triplets((current_obj,), coco_classes)
                # if current_obj not in coco_classes:
                #     continue
                if current_obj not in seen_objects_w_iou:
                    seen_objects_w_iou[current_obj] = cur_iou
                else:
                    if cur_iou > seen_objects_w_iou[current_obj]:
                        seen_objects_w_iou[current_obj] = cur_iou
            # Calculate average
            average = 0
            for obj, iou in seen_objects_w_iou.items():
                average += iou
            if len(seen_objects_w_iou) > 0:
                average /= len(seen_objects_w_iou)
                images_averages.append(average)
            print(idx)
            average = statistics.fmean(images_averages)
            print(f"Cur average: {average}")
            idx += 1
        average = statistics.fmean(images_averages)
        print(f"Average IOU: {average}")
        with open("/notebooks/avg.txt", 'w') as fout:
                json.dump(average, fout)

    def vg_experimennt_plot_hist(self, attributes_blip_path, objects_blip_path, objects_path):

        objects_blip_path = open_csv(objects_blip_path)
        attributes_blip_path = open_csv(attributes_blip_path)
        objects_data = open_csv(objects_path)

        object_blip_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in objects_blip_path[1:]]

        atrribute_blip_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in attributes_blip_path[1:]]

        object_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in objects_data[1:]]

        imgs_ids = {}

        img_ids_objs_to_ontology, img_ids_atts_to_ontology, img_ids_caps_to_ontology = {}, {}, {}
        img_ids_objs_blip_to_ontology, img_ids_atts_blip_to_ontology = {}, {}
        # Get all the image ids to the dict
        for img_id_dict in object_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            img_ids_objs_to_ontology.update({img_id: {"objects": {"clip": {"rois": list()}}}})
            img_ids_atts_to_ontology.update({img_id: {"attributes": {"clip": {"rois": list()}}}})
            img_ids_caps_to_ontology.update({img_id: {"captions": {"blip": {"rois": list()}}}})
            img_ids_objs_blip_to_ontology.update({img_id: {"objects": {"blip": {"rois": list()}}}})
            img_ids_atts_blip_to_ontology.update({img_id: {"attributes": {"blip": {"rois": list()}}}})
            if img_id not in imgs_ids:
                imgs_ids.update({img_id:''})

        # Get all the ROIs with their corresponding objects
        for img_id_dict in object_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            objects_and_confidences = list(zip(objects_data[0][:-2], img_id_dict[img_id][:-2]))
            objects_and_confidences = [(k, float(v)) for (k, v) in sorted(objects_and_confidences, key=lambda x: x[1], reverse=True)]
            roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
            img_ids_objs_to_ontology[img_id]['objects']['clip']['rois'].append({roi: objects_and_confidences})
            if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                print(img_ids_objs_to_ontology[img_id]['objects']['clip']['rois'][0][roi][:5])

        for img_id_dict in object_blip_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            objects_and_confidences = list(zip(objects_blip_path[0][:-2], img_id_dict[img_id][:-2]))
            objects_and_confidences = [(k, float(v)) for (k, v) in sorted(objects_and_confidences, key=lambda x: x[1], reverse=True)]
            roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
            img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois'].append({roi: objects_and_confidences})
            if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                print(img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois'][0][roi][:5])

        for img_id_dict in atrribute_blip_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            attributes_and_confidences = list(zip(attributes_blip_path[0][:-2], img_id_dict[img_id][:-2]))
            attributes_and_confidences = [(k, float(v)) for (k, v) in sorted(attributes_and_confidences, key=lambda x: x[1], reverse=True)]
            roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
            img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois'].append({roi: attributes_and_confidences})
            if img_id == '2316634' and roi == '[179, 405, 162, 91]':
                print(img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois'][0][roi][:5])
            

        objs_histogram = {}
        attrs_histogram = {}

        imgs_ids = list(imgs_ids.keys())
        
        for idx, img_id in enumerate(imgs_ids):

            rois_list = []
            # Add objects & attributes ROIs
            attributes_blip_rois = img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois']
            objects_blip_rois = img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois']
            objects_rois = img_ids_objs_to_ontology[img_id]['objects']['clip']['rois']
            for ijx, obj_roi in enumerate(objects_rois):
                roi = list(obj_roi.keys())[0]
                objs_confs = objects_blip_rois[ijx][roi][:10]
                attrs_confs = attributes_blip_rois[ijx][roi][:10]
                for obj_conf in objs_confs:
                    obj = obj_conf[0]
                    if obj not in objs_histogram:
                        objs_histogram[obj] = 1
                    else:
                        objs_histogram[obj] += 1

                for att_conf in attrs_confs:
                    att = att_conf[0]
                    if att not in attrs_histogram:
                        attrs_histogram[att] = 1
                    else:
                        attrs_histogram[att] += 1

        res_file = "/notebooks/objs_histogram.json"
        res_file2 = "/notebooks/atts_histogram.json"
        with open(res_file, 'w') as fout:
                json.dump(objs_histogram, fout) 
        with open(res_file2, 'w') as fout:
                json.dump(attrs_histogram, fout) 
        from matplotlib import pyplot as plt
        plt.bar(attrs_histogram.keys(), attrs_histogram.values(), width=3, color='g')
        plt.title("Attributes Histogram")
        plt.ylabel("No. of Attributes")

        plt.xticks(range(len(attrs_histogram.keys())), attrs_histogram.keys())
        plt.show()
        plt.bar(objs_histogram.keys(), objs_histogram.values(), width=3, color='g')
        plt.title("Objects Histogram")
        plt.ylabel("No. of Objects")

        plt.xticks(range(len(objs_histogram.keys())), objs_histogram.keys())
        plt.show()

    def vg_experiment_proposals_with_gt_recall2n(self):
        import itertools
        import ast
        ipc_data = json.load(open('/storage/ipc_data/paragraphs_v1.json','r'))
        sample_ids = open("/notebooks/nebula3_experiments/sample_ids_clip.txt", "r").read().splitlines()

        result_path_base = '/storage/results'
        input_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_blip_itc_60_min_w_60_ontology_vg_objects.csv")
        # input_path = "/notebooks/vg_output/results_clip_blip_itc_vg_objects_vg.json"

        print(f"Input path: {input_path}")

        objects_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_clip_60_min_w_60_ontology_vg_objects.csv")

        attributes_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_blip_itc_60_min_w_60_ontology_vg_attributes.csv")

        objects_blip_path = open_csv(input_path)
        objects_data = open_csv(objects_path)

        attributes_blip_path = open_csv(attributes_path)

        recalls_obj, recalls_attr = [], []
        dict_img_id_top_objs, dict_img_id_top_attrs = {}, {}
        object_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in objects_data[1:]]
        object_blip_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in objects_blip_path[1:]]
        atrribute_blip_image_ids = [{image_id[-1].replace("tensor([", "").replace("])",""): image_id[:-1]} for image_id in attributes_blip_path[1:]]

        imgs_ids = {}
        img_ids_objs_blip_to_ontology, img_ids_objs_to_ontology, img_ids_atts_blip_to_ontology = {}, {}, {}

        vgenome_images = '/datasets/visualgenome/VG/'
        result_path = os.path.join(os.getcwd(), 'vg_output')

        res_file = os.path.join(result_path, f"results_obj_attr_pairs_2n_vg.json")

        vlm_blip = BlipItcVlmImplementation()

        # Get all the image ids to the dict
        for img_id_dict in object_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            img_ids_objs_blip_to_ontology.update({img_id: {"objects": {"blip": {"rois": list()}}}})
            img_ids_objs_to_ontology.update({img_id: {"objects": {"clip": {"rois": list()}}}})
            img_ids_atts_blip_to_ontology.update({img_id: {"attributes": {"blip": {"rois": list()}}}})
            if img_id not in imgs_ids:
                imgs_ids.update({img_id:''})

        for img_id_dict in object_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            objects_and_confidences = list(zip(objects_data[0][:-2], img_id_dict[img_id][:-2]))
            objects_and_confidences = [(k, float(v)) for (k, v) in sorted(objects_and_confidences, key=lambda x: x[1], reverse=True)]
            roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
            img_ids_objs_to_ontology[img_id]['objects']['clip']['rois'].append({roi: objects_and_confidences})

        for img_id_dict in object_blip_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            objects_and_confidences = list(zip(objects_blip_path[0][:-2], img_id_dict[img_id][:-2]))
            objects_and_confidences = [(k, float(v)) for (k, v) in sorted(objects_and_confidences, key=lambda x: x[1], reverse=True)]
            roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
            img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois'].append({roi: objects_and_confidences})

        for img_id_dict in atrribute_blip_image_ids:
            img_id = list(img_id_dict.items())[0][0]
            attributes_and_confidences = list(zip(attributes_blip_path[0][:-2], img_id_dict[img_id][:-2]))
            attributes_and_confidences = [(k, float(v)) for (k, v) in sorted(attributes_and_confidences, key=lambda x: x[1], reverse=True)]
            roi = img_id_dict[img_id][-2].replace("tensor([", "").replace("])","")
            img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois'].append({roi: attributes_and_confidences})


        imgs_ids = list(imgs_ids.keys())

        results = {}
        print("Starting..")
        for idx, img_id in enumerate(imgs_ids):
            print(idx)
            rois_list = []
            dict_img_id_top_objs[img_id] = []
            dict_img_id_top_attrs[img_id] = []
            # Add objects & attributes ROIs
            objects_rois = img_ids_objs_to_ontology[img_id]['objects']['clip']['rois']
            objects_blip_rois = img_ids_objs_blip_to_ontology[img_id]['objects']['blip']['rois']
            attributes_blip_rois = img_ids_atts_blip_to_ontology[img_id]['attributes']['blip']['rois']
            obj_dict = dict()
            obj_dict[img_id] = dict()
            # GET TOP 20 ATTRIBUTES FOR EVERY BBOX IN THE IMAGE
            for ijx, obj_roi in enumerate(objects_rois):
                # objectss
                roi = list(obj_roi.keys())[0]
                objs_confs = objects_blip_rois[ijx][roi][:20]
                objs = [obj[0] for obj in objs_confs]
                # dict_img_id_top_objs[img_id].append(objs)
                # attributes
                attrs_confs = attributes_blip_rois[ijx][roi][:20]
                attrs = [att[0] for att in attrs_confs]
                # dict_img_id_top_attrs[img_id].append(attrs)

                objs_attrs_pairs_tuples = list(itertools.product(attrs, objs))
                objs_attrs_pairs = [' '.join(i) for i in objs_attrs_pairs_tuples]

                full_fname = os.path.join(vgenome_images, img_id + '.jpg')
                if not os.path.exists(full_fname):
                    break
                cv_img = cv2.imread(full_fname)
                if cv_img is None:
                    break
                bbox_proposal = roi
                    
                pil_img = Image.open(full_fname)
                if pil_img is None:
                    break
                bbox = ast.literal_eval(bbox_proposal)
                scores = vlm_blip.compute_similarity_on_bboxes(pil_img, objs_attrs_pairs ,bbox)
                scores_ = [(pair, scores[idx]) for idx, pair in enumerate(objs_attrs_pairs)]
                scores_sorted = sorted(scores_, key=lambda x: x[1], reverse=True)
                scores_sorted = [(score[0], str(score[1])) for score in scores_sorted]

                obj_dict[img_id].update({str(bbox_proposal) : str(scores_sorted)})
                results.update(obj_dict)
                # Intermediate save 
                with open(res_file, 'w') as fout:
                    json.dump(results, fout)        


    def compute_recall_for_pairs(self):

        from nebula3_experiments.vg_eval import VGEvaluation, get_sc_graph, spice_get_triplets, tuples_from_sg
        import os, csv
        import ast

        sample_ids = open("/notebooks/nebula3_experiments/sample_ids_clip.txt", "r").read().splitlines()

        att_pairs = json.load(open('/notebooks/vg_output/results_obj_attr_pairs_2n_vg.json','r'))
        
        ipc_data = json.load(open('/storage/ipc_data/paragraphs_v1.json','r'))

        evaluator = VGEvaluation()

        recalls_obj = []

        for sample_id in tqdm.tqdm(sample_ids):
            sample_id = int(sample_id)
            ipc = ipc_data[sample_id]
            img_id = str(ipc['image_id'])
            sg = get_sc_graph(ipc['image_id'])
            # gt_triplets = tuples_from_sg(sg)
            bbox_to_obj, bbox_to_attr = {}, {}

            for bbox_w_obj in sg.attributes:
                bbox = [bbox_w_obj.subject.x, bbox_w_obj.subject.y,
                        bbox_w_obj.subject.width, bbox_w_obj.subject.height]
                bbox = str(bbox)#str(list(bbox_xywh_to_xyxy(bbox)))
                cur_obj = bbox_w_obj.attribute #bbox_w_obj.subject.names[0]
                bbox_to_obj[bbox] = cur_obj
            if img_id in att_pairs:
                img_id_keys = list(att_pairs[img_id].keys())
                for bbox, obj in bbox_to_obj.items():
                    if bbox in img_id_keys:
                        obj_att_pairs = ast.literal_eval(att_pairs[img_id][bbox])
                        top_20_objects = []
                        for pair in obj_att_pairs:
                            obj = pair[0].split(' ')[0]
                            if obj not in top_20_objects:
                                top_20_objects.append(obj)
                        gt_obj = [bbox_to_obj[bbox]]
                        # we take top 10 objects from the top 20 objects.
                        recall = evaluator.recall_triplet_max_score(gt_obj, top_20_objects[:10])
                        recalls_obj.append(recall)
                cur_avg = np.average(recalls_obj)
                print("recall objects is: {}".format(cur_avg))

        avg_recall = np.average(recalls_obj)
        print(avg_recall)
        open("att_n_recall.txt", "w").write(str(avg_recall))






def main():
    vg_experiment = VG_EXPERIMENT()
    # vg_experiment.vg_experiment_proposals(ontology_name='vg_objects', vlm_name='blip_itc')
    # vg_experiment.compute_recall_for_pairs()
    # vg_experiment.vg_experiment_proposals_with_gt_recall2n()
    # vg_experiment.vg_experiment_add_atrs_to_proposals(ontology_name='vg_attributes', vlm_name='blip_itc')
    # vg_experiment.insert_local_proposals_tokens()
    # vg_experiment.insert_local_yolo_tokens()
    # vg_experiment.process_ofa_results()
    # vg_experiment.vg_experiment_ofa()
    # vg_experiment.vg_experiment_topdown_yolov7(ontology_name='vg_objects')
    # vg_experiment.vg_experiment_topdown_detectron(ontology_name='vg_objects')
    # vg_experiment.vg_experiment_caption(captioner='blip')
    vg_experiment.vg_experiment_topdown(ontology_name='vg_objects', vlm_name='owl_vit')
    # vg_experiment.vg_experiment_topdown_ensemble(ontology_name='vg_objects', vlm_names=['clip','blip_itc'])
    # df = pd.read_csv('/notebooks/vg_output/results_blip_itc_persons_vg.csv')
    # print(len(df))
    # a=0
    # vg_experiment.vg_experiment_topdown_paragraphs(vlm_name='blip_itc', with_split=True)
    # vg_experiment.vg_experiment_topdown_paragraphs_random(vlm_name='blip_itc', with_split=True)
    # file_path = "/notebooks/vg_output/results_blip_itc_paragraphs_vg_split.json"
    # vg_experiment.compute_average_relations(file_path)
    # file_path = "/notebooks/vg_output/results_blip_itc_paragraphs_vg_random_split.json"
    # vg_experiment.compute_average_relations(file_path)
    # vg_experiment.blip_vg_relations_experiment()
    # vg_experiment.blip_vg_relations_experiment_random()
    # vg_experiment.clip_vg_relations_experiment_random()
    # file_path = "/notebooks/vg_output/results_blip_vg_relations.json"
    # vg_experiment.compute_average_relations(file_path)
    # file_path = "/notebooks/vg_output/results_blip_vg_relations_random.json"
    # vg_experiment.compute_average_relations(file_path)
    result_path_base = '/storage/results' #os.path.join(os.getcwd(), 'vg_output')
    objects_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_clip_60_min_w_60_ontology_vg_objects.csv")
    # attributes_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_clip_60_min_w_60_ontology_vg_attributes.csv")
    objects_blip_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_blip_itc_60_min_w_60_ontology_vg_objects.csv")
    attributes_blip_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_blip_itc_60_min_w_60_ontology_vg_attributes.csv")
    # caption_path = os.path.join(result_path_base, "results_bottom_up_roi_det_vs_vg_min_h_blip_itc_60_min_w_60_ontology_fake_vg_captions.csv")
    # ipc_path = "/storage/ipc_data/paragraphs_v1.json"
    # vg_experiment.insert_vg_local_experiments_to_db(objects_path, objects_blip_path, attributes_path, attributes_blip_path, caption_path, ipc_path, db_name="nebula_playground", source='visualgenome')
    # vg_experiment.vg_experimennt_plot_hist(attributes_blip_path, objects_blip_path, objects_path)
    # result_path_base = os.path.join(os.getcwd(), 'vg_output')
    # objects_path = os.path.join(result_path_base, "results_clip_vg.csv")
    # captions_path = os.path.join(result_path_base, "results_captions_blip_vg.json")
    # persons_path = os.path.join(result_path_base, "results_blip_itc_persons_vg.csv")
    # scenes_path = os.path.join(result_path_base, "results_blip_itc_scenes_vg.csv")
    # ipc_path = "/storage/ipc_data/paragraphs_v1.json"
    # vg_experiment.insert_vg_experiments_to_db(objects_path, captions_path, persons_path, scenes_path, ipc_path, db_name="nebula_playground")
if __name__ == '__main__':
    main()