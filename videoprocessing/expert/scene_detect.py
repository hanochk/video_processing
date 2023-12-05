from posixpath import basename
from typing import Counter
from scenedetect.video_splitter import split_video_ffmpeg, is_ffmpeg_available
# Standard PySceneDetect imports:
from scenedetect import VideoManager, scene_detector
# For content-aware scene detection:
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from pickle import STACK_GLOBAL
import cv2
import os
import logging
# from arango import ArangoClient
# from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np
import glob
#import redis
import csv
import json
import sys
# sys.path.insert(0, './')
# sys.path.insert(0, 'nebula3_database/')

import torch
from PIL import Image
from movie.movie_db import MOVIE_DB
from playground.playground_db import PLAYGROUND_DB
import clip

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from expert.clip_video_utils import ClipVideoUtils

# from clip_video_utils import ClipVideoUtils
import argparse
import datetime

class SceneDetector():
    def __init__(self, debug=False):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        # self.video_eval = NebulaVideoEvaluation()
        self.adaptive_threshold = 3.0 #3.0
        self.min_delta_hsv = 15.0
        self.window_width = 2

        self.debug = debug
        if self.debug:
            self.debug_all = list()

        self.video_utils = ClipVideoUtils(batch_size=128)
        self.nre = MOVIE_DB()
        self.playground_instance = PLAYGROUND_DB()
        self.db = self.nre.db
        self.pdb = self.playground_instance.db
            # self.s3 = boto3.client('s3', region_name='eu-central-1')
        # self.use_ClipCap = use_ClipCap
        # self.use_OFA = use_OFA
        # if self.use_ClipCap:
        #     self.clip_cap = ClipCap()
        #     self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        #     self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        # if self.use_OFA:
            # self.ofa = OFA_LOADER()


    def detect_scene_elements(self, video_file, method='scene_manager'):
        print("DEBUG: method", video_file, method)
        scenes = []
        if method == 'scene_manager':
                video_manager = VideoManager([video_file])
                scene_manager = SceneManager()
                # scene_manager.add_detector(ContentDetector(threshold=30.0)) # HK was 30.0
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=self.adaptive_threshold,
                                            min_delta_hsv=self.min_delta_hsv, window_width=self.window_width)) # HK WAS 3.0
            # Improve processing speed by downscaling before processing.
                video_manager.set_downscale_factor() #No-op. Set downscale_factor in `SceneManager` instead.
            # Start the video manager and perform the scene detection.
                video_manager.start()
                scene_manager.detect_scenes(video=video_manager, show_progress=self.debug) #scene_manager.stats_manager.get_metrics(100, 'delta_hue')
                scene_list = scene_manager.get_scene_list()

                if self.debug:
                    cut_list = scene_manager.get_cut_list() ## csv: condition for new SE when adaptive_ratio>adaptive_threshold & content_val (weighted sum of delta_hue, delta_lum, delta_sat ) >min_delta_hsv
                    scene_manager.stats_manager.save_to_csv(csv_file=os.path.join('/notebooks',
                            os.path.basename(video_file).split('.')[0] + '_' +str(self.adaptive_threshold) + '_'+ str(self.window_width) + '_stats_manager.csv'),
                            base_timecode='FrameTimecode')
                    if not cut_list:
                        cut_list = [0, len(scene_manager.stats_manager._frame_metrics.items())]
                    print("cut_list {}".format(cut_list))
                    self.debug_all.append({os.path.basename(video_file).split('.')[0]: cut_list})

                if not scene_list:
                    scene = [0, len(scene_manager.stats_manager._frame_metrics.items())]
                    if self.debug:
                        print("All movi 1 Scenes element: ", scenes)
                    scenes.append([scene[0], scene[1]])
                else:
                    for i, scene in enumerate(scene_list):
                        start_frame = scene[0].get_frames()
                        stop_frame = scene[1].get_frames()
                        scenes.append([start_frame,stop_frame])
                    # secnes is list of lists
                print("Scenes elements: ", scenes)
        elif method == 'clip':
            # boundaries is list of tuples
            boundaries, good_frame_len = self.video_utils.get_scene_elements_from_embeddings(video_file)
            scenes = []
            for boundary in boundaries:
                scenes.append([boundary[0], boundary[1]])
            print("Scenes elements: ", scenes)

        return(scenes)

    def divide_movie_into_frames(self, movie_in_path, movie_out_folder, save_frames=False):
        """
        Input: Diretory path of the movies and directory of where to save the frames.
        Output: frames (and if save_frames=True it also saves in the destination)
        """
        frames = []
        cap = cv2.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        frames.append(frame)
        num = 0
        if save_frames:
            cv2.imwrite(os.path.join(movie_out_folder, f'frame{num:04}.jpg'), frame)
        while cap.isOpened() and ret:
            num = num + 1
            ret, frame = cap.read()
            frames.append(frame)
            if frame is not None and save_frames:
                cv2.imwrite(os.path.join(movie_out_folder,
                           f'frame{num:04}.jpg'), frame)
        return frames

    # def store_frames_to_s3(self, movie_id, frames_folder, video_file):
    #     bucket_name = "nebula-frames"
    #     folder_name = movie_id
    #     self.s3.put_object(Bucket=bucket_name, Key=(folder_name+'/'))
    #     print(frames_folder)
    #     if not os.path.exists(frames_folder):
    #         os.mkdir(frames_folder)
    #     else:
    #         for f in os.listdir(frames_folder):
    #             if os.path.isfile(os.path.join(frames_folder, f)):
    #                 os.remove(os.path.join(frames_folder, f))
    #     num_frames = self.divide_movie_into_frames(video_file, frames_folder)
    #     # SAVE TO REDIS - TBD
    #     if num_frames > 0:
    #         for k in range(num_frames):
    #             img_name = os.path.join(
    #                 frames_folder, f'frame{k:04}.jpg')
    #             self.s3.upload_file(img_name, bucket_name, folder_name +
    #                         '/' + f'frame{k:04}.jpg')

    def get_video_metadata(self, video_file):
        """
        Input: video
        Output: width, height and fps of the video.
        """
        cap = cv2.VideoCapture(video_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return ({'width': W, 'height': H, 'fps': fps})

    def detect_scenes(self, video_file):
        """
        Input: video
        Output: Scenes of the video.
        """
        scenes = []

        video_manager = VideoManager([video_file])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=70.0))
    # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            stop_frame = scene[1].get_frames()
            scenes.append([start_frame,stop_frame])
        print("Scenes: ", scenes)
        return(scenes)
    
    def get_mdf_every_n_sec(self, video_file, n_sec):
        mdfs = []
        mdfs = self.video_utils.get_frame_every_n_sec(video_file, n_sec=n_sec)
        return mdfs
    
    def detect_mdf(self, video_file, scene_elements, method='meanshift'):
        """
        :param video_file:
        :param scene_elements: list of list, [[2, 4], [33, 55]] - start and end frame of each scene
        :param method: '3frames' - choose three good frames, 'clip_segment' - use clip segmentation to choose 3 MDFs
        :return: list of lists
        """
        print("Detecting MDFs..")
        if method == '1frame':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                start_frame = scene_element[0]
                stop_frame = scene_element[1]
                blur_threshold, values, fft_center, fft_border = self.video_utils.get_adaptive_movie_threshold(video_file,
                    start_frame, stop_frame)
                # Don't take the first or the last frame
                # values[0] = values[-1] = -1
                scene_mdfs.append(int(np.argmax(values) + scene_element[0]))
                mdfs.append(scene_mdfs)
            pass
        elif method == 'fft_center':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                start_frame = scene_element[0]
                stop_frame = scene_element[1]
                blur_threshold, values, fft_center, fft_border = self.video_utils.get_adaptive_movie_threshold(
                    video_file,
                    start_frame, stop_frame)
                scene_mdfs.append(int(np.argmin(fft_center) + scene_element[0]))
                mdfs.append(scene_mdfs)
        elif method == 'fft_border':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                start_frame = scene_element[0]
                stop_frame = scene_element[1]
                blur_threshold, values, fft_center, fft_border = self.video_utils.get_adaptive_movie_threshold(
                    video_file,
                    start_frame, stop_frame)
                scene_mdfs.append(int(np.argmax(fft_border) + scene_element[0]))
                mdfs.append(scene_mdfs)
        elif method == '3frames':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                start_frame = scene_element[0]
                stop_frame = scene_element[1]
                # frame_qual = self.video_eval.mark_blurred_frames(video_file, start_frame, stop_frame, -1)
                frame_qual = self.video_utils.mark_blurred_frames(video_file, start_frame, stop_frame, -1)

                # Ignore the blurred images
                frame_qual[0:3] = 0
                frame_qual[-3:] = 0
                middle_frame = start_frame + ((stop_frame - start_frame) // 2)
                good_frames = np.where(frame_qual > 0)[0]
                if len(good_frames > 5):
                    stop_frame = start_frame + good_frames[-1]
                    middle_frame = start_frame + good_frames[len(good_frames) // 2]
                    start_frame = start_frame + good_frames[0]
                    scene_mdfs.append(int(start_frame))
                    scene_mdfs.append(int(middle_frame))
                    scene_mdfs.append(int(stop_frame))
                else:
                    scene_mdfs.append(int(start_frame) + 2)
                    scene_mdfs.append(int(middle_frame))
                    scene_mdfs.append(int(stop_frame) - 2)
                mdfs.append(scene_mdfs)
        elif method == 'clip_segment':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                chosen_mdf, ret_img = self.video_utils.choose_best_frame(video_file,
                    scene_element[0], scene_element[1])
                scene_mdfs.append(int(chosen_mdf))
            mdfs.append(scene_mdfs)
        elif method == 'meanshift':
            mdfs = []
            ret_imgs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                return_img_list = []
                chosen_mdf, ret_img, cluster_size = self.video_utils.choose_frames_with_meanshift(video_file,
                    scene_element[0], scene_element[1])
                sorted_by_size = list(np.argsort(cluster_size)[::-1])
                for k in sorted_by_size:
                    scene_mdfs.append(int(chosen_mdf[k]))
                    return_img_list.append(ret_img[k])
                ret_imgs.append(return_img_list)
                mdfs.append(scene_mdfs)
        else:
            raise Exception('Unsupported method')
        # mdfs = []
        # for scene_element in scene_elements:
        #     scene_mdfs = []
        #     start_frame = scene_element[0]
        #     stop_frame = scene_element[1]
        #     scene_mdfs.append(start_frame + 2)
        #     middle_frame = start_frame + ((stop_frame- start_frame) // 2)
        #     scene_mdfs.append(middle_frame)
        #     scene_mdfs.append(stop_frame - 2)
        #     mdfs.append(scene_mdfs)
        return (mdfs)


    def save_mdfs_to_jpg(self, full_path, mdfs, save_path="/dataset/lsmdc/mdfs_of_20_clips/"):
        vidcap = cv2.VideoCapture(full_path)
        success, image = vidcap.read()
        count = 0
        counted_mdfs = 0
        img_names = []
        mdfs = list(set(mdfs)) #remove duplicate mdfs
        if ".mp4" in full_path:
            full_path_modified = full_path.split("/")[-1].replace(".mp4","")
        if ".avi" in full_path:
            full_path_modified = full_path.split("/")[-1].replace(".avi","")
        while success:
            if count in mdfs and success:
                img_name = f"{save_path}{full_path_modified}__{count}.jpg"
                img_names.append(img_name)
                cv2.imwrite(img_name, image)     # save frame as JPEG file
                print('Read a new frame: ', success)
                counted_mdfs += 1
            if counted_mdfs == len(mdfs):
                print("Done")
                return img_names
            count += 1
            # exit of infinite loop
            if count > 9999:
                return img_names
            success, image = vidcap.read() # Frame #0 was read at the begining
            
    def generate_captions_on_mdfs(self, images, mdfs_path, prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/dataset1/mdfs/"
     ):
        # images_lst = [os.path.join(mdfs_path, image) for image in os.listdir(mdfs_path) if image in images]
        output_dict = {}
        for img_path in images:
            img_name = img_path.split("/")[-1]
            # CLIP CAP
            if self.use_ClipCap:
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = self.clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
                embedding = self.clip_model.encode_image(img)
                clip_cap_output = self.clip_cap.generate_text(embedding, use_beam_search=False)
                output_dict[img_name] = [clip_cap_output]

            # OFA
            if self.use_OFA:
                image = Image.open(img_path)
                sample = self.ofa.construct_sample(image)
                result, _ = self.ofa.evaluate_caption(sample)
                ofa_output = result[0]['caption']
                if not self.use_ClipCap:
                    # ClipCap wasn't used, we use only one string caption
                    output_dict[img_name] = ofa_output
                else:
                    # ClipCap was used, so we have a list here.
                    output_dict[img_name].append(ofa_output)
        return output_dict


    def convert_avi_to_mp4(self, avi_file_path, output_name):
        os.system("ffmpeg -y -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
            input=avi_file_path, output=output_name))
        return True

    # file_name - file name without path
    # movie_name - if available, if no - same file name
    # tags - if available
    # full_path - path to file, for video processing
    # url - url to S3
    # last_frame - number of frames in movie
    # metadata - available metadata - fps, resolution....
    def insert_movie(self, full_path, url, source, db_name="prodemo"):
        # Change Database to `ilan_test`
        self.nre.change_db("prodemo")
        self.db = self.nre.db
        file_name = full_path.split("/")[-1].replace(".mp4", "")
        movie_id = self.nre.get_movie_by_filename(file_name)
        query = 'UPSERT { File: @File, scene_element: @scene_element} INSERT  \
            { File: @File, scene_element: @scene_element, \
                url_path: @url, \
                 movie_id: @movie_id, captions: @captions, ref: @ref, experts: @experts, groundtruth: @groundtruth, base: @base,\
                         source: "lsmdc_concatenated"\
                    } UPDATE \
                { File: @File, scene_element: @scene_element, url_path: @url, base: @base,\
                    movie_id: @movie_id, captions: @captions, source: @source, ref: @ref, experts: @experts \
                } IN s2_clsmdc \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        scene_elements = self.detect_scene_elements(full_path)
        for idx_scene_element, scene_element in enumerate(scene_elements):
            mdfs = self.detect_mdf(full_path, [scene_element], method='meanshift')
            mdf_outputs = []
            img_caption_outputs = {}
            for mdf in mdfs:
                mdf_images_path = self.save_mdfs_to_jpg(full_path, mdf, save_path="/dataset/lsmdc/mdfs_of_20_clips/")
                if mdf_images_path:
                    caption_outputs = self.generate_captions_on_mdfs(mdf_images_path,
                            mdfs_path="/dataset/lsmdc/mdfs_of_20_clips/",
                            prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/dataset1/mdfs/")
                    img_caption_outputs.update(caption_outputs)


            bind_vars = {
                            'ref': '',
                            'movie_id': movie_id,
                            'scene_element': idx_scene_element,
                            'File': file_name,
                            'url': url,
                            'base': '',
                            'captions': img_caption_outputs,
                            'source': source,
                            'experts': {},
                            'groundtruth': []
                        }
            print(bind_vars)
            # Here we are using the PLAYGROUND database (self.pdb)
            self.pdb.aql.execute(query, bind_vars=bind_vars)

    def update_s1_lsmdc_with_ofa(self):
        #@TODO: update playground_db with a new function get_document
        doc_s1_lsmdc = self.playground_instance.get_document(document_name="s1_lsmdc")
        for movie in doc_s1_lsmdc:
            print(movie)

            query = 'UPSERT { movie_id: @movie_id} INSERT  \
                { ofa: @ofa, url: @url, actions: @actions, base: @base, \
                    places: @places, experts: @experts, groundtruth: @groundtruth, \
                    scene_element: @scene_element, movie_id: @movie_id, ref: @ref, \
                       _key: @_key, _id: @_id, _rev: @_rev\
                    } UPDATE \
                {   ofa: @ofa \
                } IN s1_lsmdc'

            movie_id = movie['movie_id']
            scene_element_idx = movie['scene_element']
            scene_element = self.playground_instance.get_movie_info(movie_id=movie_id)[0]['scene_elements'][scene_element_idx]
            full_path = os.path.join("/dataset/development",movie['url'].split("/")[-1])
            mdfs = self.detect_mdf(full_path, [scene_element], method='meanshift')
            mdf_outputs = []
            img_caption_outputs = {}
            for mdf in mdfs:
                mdf_images_path = self.save_mdfs_to_jpg(full_path, mdf, save_path="/dataset/lsmdc/mdfs_of_s1_lsmdc/")
                if mdf_images_path:
                    caption_outputs = self.generate_captions_on_mdfs(mdf_images_path,
                            mdfs_path="/dataset/lsmdc/mdfs_of_s1_lsmdc/",
                            prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/dataset1/mdfs/")
                    img_caption_outputs.update(caption_outputs)

            movie["ofa"] = img_caption_outputs
            self.pdb.aql.execute(query, bind_vars=movie)


    def get_movie_id_from_url(self, url):
        movie_id = url.split("/")[-1].replace(".","_")
        if "_mp4" in movie_id:
            return movie_id.replace("_mp4", "")
        elif "_avi" in movie_id:
            return movie_id.replace("_avi", "")
        else:
            raise Exception("Unknown video format. Please use mp4/avi")


    def new_movies_batch_processing(self, upload_dir, storage_dir, dataset):
        _files = glob.glob(upload_dir + '/*')
        movies = []
        for _file in _files:
            file_name = basename(_file)
            movie_mame = file_name.split(".")[0]
            print("New movie found")
            print(file_name)
            print(movie_mame)
            self.convert_avi_to_mp4(_file, storage_dir + "/" + movie_mame)
            movie_id = self.insert_movie(file_name, movie_mame, ["lsmdc", "pegasus", "visual genome"],
                              storage_dir + "/" + movie_mame + ".mp4", "static/" + dataset + "/" + movie_mame + ".mp4", 300, "created")
            movies.append((_file, movie_id))
        return(movies)

    def divide_movie_by_timestamp(self, movie_in_path, t_start, t_end, dim=(224, 224)):

        cap = cv2.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        num = 0
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        movie_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_start = t_start * fps
        frame_end = t_end * fps
        if frame_start > movie_frames_count or frame_end > movie_frames_count:
            print(f'Error, the provided t_start {t_start} or t_end {t_end} \
                multiplied by fps {fps} is higher than number of frames {movie_frames_count}')
            return -1
        while cap.isOpened() and ret:
            num = num + 1
            ret, frame = cap.read()
            if frame is not None and num >= frame_start and num <= frame_end:
                resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                frames.append(resized_frame)
        return frames

    def preprocess_dataset(self, dataset_path, db_type="COMET"):
        """
        Preprocess LSMDC & VSCOMET datasets to dictionaries
        """
        new_db = []
        if db_type == "COMET":
            test_json = open(dataset_path)
            data = json.load(test_json)
            for idx, key in enumerate(data):
                if "lsmdc" in key['img_fn']:
                    new_key = self.get_frame_dict(key)
                    new_db.append(new_key)

        elif db_type == "LSMDC":
            with open(dataset_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    key = row[0].split()[0]
                    txt = ''
                    if len(row) > 1:
                        txt = row[1]
                    new_db.append({'clip_id': key, 'txt': txt})
            return new_db
        else: #db_type is LSMDC
            print("Unknown DB-TYPE")

        return new_db

    def time2secs(self, ts):
        hh, mm, ss, ms = ts.split(".")
        return int(hh)*3600 + int(mm)*60 + int(ss) + (float(ms) / 1000000)

    def get_gt_clips(self, path):
        output = []
        with open(path) as fp:
            lines = fp.readlines()
            cnt = 1
            for line in lines:
                output.append(line.rstrip("\n"))
        return output

    def divide_to_parts(self, dataset, clip_time=20, only_gt_clips=False):
        """
         Divide movies to 20 second clips. (With holes)
         If `only_gt_clips=True`, then we take the 44 ground-truth clips only.
        """
        new_dataset = [[]]
        time_ctr = clip_time
        new_dataset_idx = 0
        start_merging = False
        counter = 0
        if only_gt_clips:
            # clip_names = self.get_playground_movies()
            clips_gt_list_path = "./utils/44_clips_list.txt"
            clip_names = self.get_gt_clips(clips_gt_list_path)
            start_merging = False
        for i in range(1, len(dataset)):
            cur_clip = dataset[i]
            clip_id = cur_clip['clip_id']
            # If this is true, we'll process only 44 clips instead of all the clips.
            if only_gt_clips and not start_merging:
                if clip_id.replace(".","_") not in clip_names:
                    continue
                else:
                    start_merging = True
                    time_ctr = clip_time
                    counter += 1
                    print(clip_id.replace(".","_"))
            ts_start, ts_end = clip_id.split("_")[-1].split("-")
            ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
            clip_length = ts_end - ts_start

            # clip is over `clip_time=20seconds`, add it alone
            if clip_length > clip_time:
                new_dataset[new_dataset_idx].append(cur_clip)
                time_ctr = clip_time
                new_dataset.append([])
                new_dataset_idx += 1
                if start_merging:
                    start_merging = False
                continue

            # clip is less than 20 seconds, we add it to new list that can contain
            # more clips in that list.
            if time_ctr - clip_length > 0:
                new_dataset[new_dataset_idx].append(cur_clip)
                time_ctr -= clip_length
                continue

            # If we add a clip that exceeds `time_ctr` of current list of clips
            # we add it to a new list.
            if time_ctr - clip_length <= 0:
                if start_merging:
                    start_merging = False
                    time_ctr = clip_time
                    new_dataset_idx += 1
                    new_dataset.append([])
                    continue
                new_dataset.append([])
                new_dataset_idx += 1
                new_dataset[new_dataset_idx].append(cur_clip)
                time_ctr = clip_time
                # If the added clip has bigger value then clip_time, we create new level
                if clip_length > clip_time:
                    new_dataset.append([])
                    new_dataset_idx += 1
                # just substract clip_length in the new level.
                else:
                    time_ctr -= clip_length
                continue
        print(counter)
        # Sometimes the last element in the list is empty, so we remove i
        if not new_dataset[-1]:
            return new_dataset[:-1]
        return new_dataset

    def remove_holes_in_concatenated_clips(self, dataset, delta_error=1):
        """
         There are clips that we concatenate that are not sequential.
         For e.g. if clip starts at 00:01:00 and ends at 00:01:20, the next clip,
         should start at approximately 00:01:20 as well, if it starts at 00:02:00 it's bad,
         so we will skip it.
         `delta_error`: maximum time allowed (in seconds) between end of clip and start of new one.
        """
        new_dataset = []
        for data in dataset:
            time_stamps = []
            for i in range(len(data)):
                 ts_start, ts_end = data[i]['clip_id'].split("_")[-1].split("-")
                 ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
                 time_stamps.append([ts_start, ts_end])

            if len(time_stamps) == 1:
                new_dataset.append(data)
                ts_start, ts_end = data[0]['clip_id'].split("_")[-1].split("-")
                ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
                length = ts_end - ts_start
                # print("-"*30)
                # print(f"Movie Name: {data[0]['clip_id']}")
                # print(f"Length: {length}")
            else:
                length = 0
                for i in range(len(time_stamps) - 1):
                    delta = time_stamps[i + 1][0] - time_stamps[i][1]
                    if delta > delta_error:
                        position = i
                        break
                if position != 0:
                    new_dataset.append(data[:position + 1])

                for i in range(len(data[:position + 1])):
                     ts_start, ts_end = data[i]['clip_id'].split("_")[-1].split("-")
                     ts_start, ts_end = self.time2secs(ts_start), self.time2secs(ts_end)
                     length += ts_end - ts_start
                if position != 0:
                    print("-"*30)
                    print(f"Movie Name: {data[0]['clip_id']}")
                    print(f"Length: {length}")
        print(f"Length of dataset: {len(new_dataset)}")

        # Sort the database by the length in a descending order, therefore we will get the longest videos w/o holes.
        new_dataset.sort(key=len, reverse=True)
        return new_dataset

    def split_modified(self, strng, sep, pos):
        """
        There're movies like Spider-man...timestamp-timestamp..
        I want to split only by the last "-"
        """
        strng = strng.split(sep)
        return sep.join(strng[:pos]), sep.join(strng[pos:])

    def check_if_merged_mp4_exists(self, path, movie_names):
        """
        For debugging purposes. Skipping merged .mp4 files.
        """
        from natsort import natsorted
        if not movie_names:
            return
        files = natsorted(movie_names)
        # Beginning timestamp of first clip, and end timestamp of last clip.
        prefix_str = self.split_modified(files[0].split("/")[-1], "-", -1)[0]
        unique_clip_name = prefix_str + "-" + \
                            files[-1].split("/")[-1].split("-")[-1].replace(".avi", ".mp4")
        if unique_clip_name in os.listdir(path):
            print(f"Merged clip {unique_clip_name} already exists! Skipping...")
            return True
        return False

    def process_specific_avi_to_mp4(self, upload_dir, storage_dir, clips_names = ['']):
        """
        Convert LSMDC .avi clips to .mp4 clips.
        If `clips_names` is [''] we return.
        Otherwise, we iterate only over the `clips_names` list.
        """
        print("Processing .avi files.")
        if clips_names != ['']:
            print(f"We iterate only over {clips_names}")
        else:
            print("Error! `clip_names` is empty.")
            return
        # Check if .mp4 and doesn't exist. Otherwise, we proceed and convert .avi to .mp4
        for clip in clips_names:
            if clip.replace(".avi",".mp4") in os.listdir(storage_dir):
                print("Skipping. Found at least one clip in folder.")
                return
        # check if merged .mp4 doesn't exist.
        if self.check_if_merged_mp4_exists(storage_dir, clips_names):
            print("Skipping. Found the merged .mp4 in folder.")
            return

        _dirs = glob.glob(upload_dir + '/*')
        movies_output = []
        counter = 0
        length_clips = len(clips_names)
        for _dir in _dirs:
            _files = glob.glob(_dir + '/*')
            for _file in _files:
                file_name = basename(_file)
                movie_name = file_name
                if movie_name in clips_names:
                    self.convert_avi_to_mp4(_file , storage_dir + "/" + movie_name.replace(".avi",""))
                    movies_output.append(storage_dir + "/" + movie_name.replace(".avi",".mp4"))
        return movies_output

    def merge_multiple_mp4_to_one(self, path, movie_names):
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        import os
        from natsort import natsorted
        # check if merged .mp4 doesn't exist.
        if not movie_names:
            print("Skipping. no `movie_names`.")
            return
        if len(movie_names) == 1:
            print(f"Skipping. We have only one {movie_names} .mp4 file, so we don't need to merge it.")
            return
        print("Merging mp4 clips to one mp4 clip..")
        L =[]
        _files = [os.path.join(path, f) for f in os.listdir(path) if f.replace(".mp4",".avi") in movie_names]
        files = natsorted(_files)
        # Beginning timestamp of first clip, and end timestamp of last clip.
        prefix_str = self.split_modified(files[0].split("/")[-1], "-", -1)[0]
        unique_clip_name = prefix_str + "-" + \
                            files[-1].split("/")[-1].split("-")[-1]
        if unique_clip_name in path:
            print(f"Merged clip {unique_clip_name} already exists! Skipping...")
        for file in files:
            if file != unique_clip_name:
                if os.path.splitext(file)[1] == '.mp4':
                    filePath = os.path.join(path, file)
                    video = VideoFileClip(filePath)
                    L.append(video)

        final_clip = concatenate_videoclips(L)
        final_clip.to_videofile(os.path.join(path,unique_clip_name), fps=24, remove_temp=False)
        print(f"Done merge the clips! Merged file: {os.path.join(path,unique_clip_name)}")
        prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/"
        print(f"Can be access in: {os.path.join(os.path.join(prefix_link, path) ,unique_clip_name)}")


    def concat_mp4(self, clips_dict, base_path = "/dataset1/", dest_path = "/dataset1/concatenated_mp4_50"):
        """
        Merging multiple mp4 files into one mp4 files.
        1. Take the relevant .avi files.
        2. Convert them to .mp4 files and save them.
        3. Merge these files to one output mp4 file.
        """
        for ctr, clips in enumerate(clips_dict):
            # 1. Take the relevant .avi files.
            clip_names = [clip['clip_id']+".avi" for clip in clips]
            # 2. Convert them to .mp4 files and save them.
            movies_output = self.process_specific_avi_to_mp4(base_path, dest_path, clip_names)
            # 3. Merge these files to one output mp4 file.
            if movies_output:
                self.merge_multiple_mp4_to_one(dest_path, clip_names)
            print(f"--------Finished {ctr+1}/{len(clips_dict)} clips.-------")

            print("Removing all the merged .mp4 files and moving to next clip.")
            clips_to_del = [clip['clip_id']+".mp4" for clip in clips]
            if len(clips_to_del) > 1: # If it's one clip, it's also the .merged one so we skip it
                for clip_to_del in clips_to_del:
                    ## If file exists, delete it.
                    clip_name = os.path.join(dest_path, clip_to_del)
                    if os.path.isfile(clip_name) and ".mp4" in clip_name:
                        os.remove(clip_name)
                    else:
                        print("Error: %s file not found" % clip_name)

    def create_concat_lsmdc_movies(self, lsmdc_dir, num_of_movies=55, base_path = "/dataset1/", dest_path = "/dataset1/concatenated_mp4_20"):

        # Paths to LSMDC .JSONs
        lsmdc_train_path = os.path.join(lsmdc_dir, "LSMDC16_annos_training.csv")
        lsmdc_val_path = os.path.join(lsmdc_dir, "LSMDC16_annos_val.csv")
        lsmdc_test_path = os.path.join(lsmdc_dir, "LSMDC16_annos_test.csv")

        # Process LSMDC annotation files
        lsmdc_train_dict = self.preprocess_dataset(lsmdc_train_path, "LSMDC")
        lsmdc_val_dict = self.preprocess_dataset(lsmdc_val_path, "LSMDC")
        lsmdc_test_dict = self.preprocess_dataset(lsmdc_test_path, "LSMDC")

        # Divide LSMDC annotation files with holes.
        lsmdc_train_dict_merged = self.divide_to_parts(lsmdc_train_dict, clip_time=20, only_gt_clips=False)
        # lsmdc_val_dict_merged = self.divide_to_parts(lsmdc_val_dict, clip_time=20, only_gt_clips=False)
        # lsmdc_test_dict_merged = self.divide_to_parts(lsmdc_test_dict, clip_time=20, only_gt_clips=False)

        # Divide LSMDC annotation files without holes.
        lsmdc_train_dict_merged = self.remove_holes_in_concatenated_clips(lsmdc_train_dict_merged)
        # lsmdc_val_dict_merged = self.remove_holes_in_concatenated_clips(lsmdc_val_dict_merged)
        # lsmdc_test_dict_merged = self.remove_holes_in_concatenated_clips(lsmdc_test_dict_merged)

        self.concat_mp4(lsmdc_train_dict_merged[:num_of_movies + 1], base_path, dest_path)


def create_mdf_string_save_img(method, scene_element, file_name, scene_detector, scene, img_folder):
    mdf_string = ''
    mdf_list = scene_detector.detect_mdf(file_name, [scene_element], method)
    for mdfs in mdf_list:
        mdfs = list(set(mdfs))
        for mdf in mdfs:
            mdf_string = mdf_string + ' ' + str(mdf)

    mdf_images = scene_detector.video_utils.get_specific_frames(file_name, mdf_list)

    k = 0
    for mdf_set in mdf_list:
        if scene_element[1] in mdf_set:
            mdf_set.remove(scene_element[1])
        mdf_set = list(set(mdf_set))
        for mdf in mdf_set:
            imgname = scene[scene.find('/') + 1:scene.rfind('.')] + f'_old_scene_{mdf:04}.jpg'
            cv2.imwrite(os.path.join(img_folder, imgname), mdf_images[k])
            k = k + 1

    return mdf_string

def paperspace_playground_mk():
    # base_folder = '/notebooks/data/videos'
    # out_folder = '/notebooks/data/mdfs'
    base_folder = '/media/media/movies/american_beauty'
    out_folder = '/media/media/movies/american_beauty'#'/home/hanoch/mdf_lsmdc/all'
    
    videos = os.listdir(base_folder)
    videos = sorted(videos)
    scene_detector = SceneDetector()
    out_file_csv = open(os.path.join(out_folder, 'result_csv7.csv'), 'w')
    fieldnames = ['movie name', 'scenes_adaptive', 'mdf_clip', 'time_per_frame']
    writer = csv.DictWriter(out_file_csv, fieldnames=fieldnames)
    writer.writerow({'movie name': 'movie name',
                     'scenes_adaptive': 'scenes_adaptive',
                     'mdf_clip': 'mdf_clip',
                     'time_per_frame': 'time_per_frame'})

    # Create LSMDC videos
    base_folder = '/lsmdc_dataset/'
    folders = os.listdir(base_folder)
    max_folders = 10
    cnt = 0
    videos = []
    for folder in folders:
        if not os.path.isdir(os.path.join(base_folder, folder)):
            continue
        video_files = os.listdir(os.path.join(base_folder, folder))
        for video_file in video_files:
            videos.append(os.path.join(base_folder, folder, video_file))

        cnt = cnt + 1
        if cnt >= max_folders:
            break

    multi_folder = 1

    for video in videos:
        # if video != 'video8188.mp4':
        #     continue
        start_time = datetime.datetime.now()
        if multi_folder == 0:
            scene_elements_adaptive = scene_detector.detect_scene_elements(os.path.join(base_folder, video))
        else:
            scene_elements_adaptive = scene_detector.detect_scene_elements(video)
        adaptive_diff = (datetime.datetime.now() - start_time).total_seconds()

        # start_time = datetime.datetime.now()
        # scene_elements_clip = scene_detector.detect_scene_elements(os.path.join(base_folder, video), method='clip')
        # clip_diff = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        if multi_folder == 0:
            old_clip_mdfs = scene_detector.detect_mdf(os.path.join(base_folder, video), scene_elements_adaptive,
                                                  method='meanshift')
        else:
            old_clip_mdfs = scene_detector.detect_mdf(video, scene_elements_adaptive, method='meanshift')
        clip_diff = (datetime.datetime.now() - start_time).total_seconds()

        num_frames = scene_elements_adaptive[-1][-1]

        adaptive_string = ''
        for scene_elem in scene_elements_adaptive:
            adaptive_string = adaptive_string + ' ' + str(scene_elem[0]) + '-' + str(scene_elem[1]) + ','

        mdf_clip_string = ''
        save_mdfs = []
        for mdfs in old_clip_mdfs:
            for mdf in mdfs:
                mdf_clip_string = mdf_clip_string + ' ' + str(mdf) + ','
                save_mdfs.append(mdf)
            mdf_clip_string = mdf_clip_string + ';'

        save_mdfs.sort()
        if multi_folder == 0:
            frames_to_save = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, video), [save_mdfs])
        else:
            frames_to_save = scene_detector.video_utils.get_specific_frames(video, [save_mdfs])

        if multi_folder == 0:
            new_video = video
        else:
            new_video = video[video.rfind('/') + 1:]

        for k, frame in enumerate(frames_to_save):
            if multi_folder == 0:
                new_name = new_video[:new_video.find('.')]
            else:
                new_name = new_video[:new_video.rfind('.')]
            imgname = new_name + f'_clipmdf_{save_mdfs[k]:04}.jpg'
            ret = cv2.imwrite(os.path.join(out_folder, imgname), frame)

        writer.writerow({'movie name': new_name,
                         'scenes_adaptive': adaptive_string,
                         'mdf_clip': mdf_clip_string,
                         'time_per_frame': f'{clip_diff/num_frames:4.3f}'})

def paperspace_playground():
    # base_folder = '/datasets/msrvtt'
    base_folder = '/datasets/dataset/msr_vtt/videos' #'/home/paperspace/data/videos'
    outfolder = '/notebooks/nebula3_playground' #'/datasets/dataset/msr_vtt/results'

    videos = os.listdir(base_folder)
    videos = sorted(videos)

    scene_detector = NEBULA_SCENE_DETECTOR(use_ClipCap=False, use_OFA=False, use_nebula3=False)

    out_file_csv = open(os.path.join(outfolder, 'result_csv2.csv'), 'w')
    fieldnames = ['movie name', 'scenes_adaptive', 'scenes_clip', 'time_adaptive', 'time_clip', 'mdf_clip', 'mdf_laplacian',
                  'time_mdf_clip', 'time_mdf_laplacian']
    writer = csv.DictWriter(out_file_csv, fieldnames=fieldnames)
    writer.writerow({'movie name': 'movie name',
                     'scenes_adaptive': 'scenes_adaptive',
                     'scenes_clip': 'scenes_clip',
                     'time_adaptive': 'time_adaptive',
                     'time_clip': 'time_clip',
                     'mdf_clip': 'mdf_clip',
                     'mdf_laplacian': 'mdf_laplacian',
                     'time_mdf_clip': 'time_mdf_clip',
                     'time_mdf_laplacian': 'time_mdf_laplacian'})

    for video in videos:
        start_time = datetime.datetime.now()
        scene_elements_adaptive = scene_detector.detect_scene_elements(os.path.join(base_folder, video))
        adaptive_diff = (datetime.datetime.now() - start_time).total_seconds()
        start_time = datetime.datetime.now()
        scene_elements_clip = scene_detector.detect_scene_elements(os.path.join(base_folder, video), method='clip')
        clip_diff = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        frame1_mdfs = scene_detector.detect_mdf(os.path.join(base_folder, video), scene_elements_adaptive, method='1frame')
        frame1_mdf_diff = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        old_clip_mdfs = scene_detector.detect_mdf(os.path.join(base_folder, video), scene_elements_adaptive, method='meanshift')
        clip_mdfs = []
        for mdf in old_clip_mdfs:
            clip_mdfs.append([mdf[0]])

        clip_mdf_diff = (datetime.datetime.now() - start_time).total_seconds()

        frames_to_save = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, video), clip_mdfs)
        for k, frame in enumerate(frames_to_save):
            imgname = video[:video.find('.')] + f'_clipmdf_{clip_mdfs[k][0]:04}.jpg'
            ret = cv2.imwrite(os.path.join(outfolder, imgname), frame)


        frames_to_save = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, video), frame1_mdfs)
        for k, frame in enumerate(frames_to_save):
            imgname = video[:video.find('.')] + f'_laplace_{frame1_mdfs[k][0]:04}.jpg'
            ret = cv2.imwrite(os.path.join(outfolder, imgname), frame)

        adaptive_string = ''
        for scene_elem in scene_elements_adaptive:
            adaptive_string = adaptive_string + ' ' + str(scene_elem[0]) + '-' + str(scene_elem[1]) + ','
        clip_string = ''
        for scene_elem in scene_elements_clip:
            clip_string = clip_string + ' ' + str(scene_elem[0]) + '-' + str(scene_elem[1]) + ','

        mdf_laplace_string = ''
        for mdf in frame1_mdfs:
            mdf_laplace_string = mdf_laplace_string + ' ' + str(mdf[0]) + ','

        mdf_clip_string = ''
        for mdf in clip_mdfs:
            mdf_clip_string = mdf_clip_string + ' ' + str(mdf[0]) + ','

        fieldnames = ['movie name', 'scenes_adaptive', 'scenes_clip', 'time_adaptive', 'time_clip', 'mdf_clip',
                      'mdf_laplacian',
                      'time_mdf_clip', 'time_mdf_laplacian']
        writer.writerow({'movie name': video[:video.find('.')],
                         'scenes_adaptive': adaptive_string,
                         'scenes_clip': clip_string,
                         'time_adaptive': str(adaptive_diff),
                         'time_clip': str(clip_diff),
                         'mdf_clip': mdf_clip_string,
                        'mdf_laplacian': mdf_laplace_string,
                        'time_mdf_clip': str(clip_mdf_diff),
                         'time_mdf_laplacian': str(frame1_mdf_diff)})

    if scene_detector.debug:
        import pandas as pd
        df = pd.DataFrame(scene_detector.debug_all)
        df = df.transpose()
        df.to_csv(os.path.join(outfolder, 'se_list' + str(scene_detector.adaptive_threshold) + '.csv'))


def test_different_thresholds():
    base_folder = '/notebooks/dataset' #'/home/hanoch/dataset/msr_vtt'
    # scenes = ["2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_01.03.14.878-01.03.15.557.avi",
    #         "2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_00.06.36.732-00.06.38.382.avi",
    #         "2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_00.38.04.753-00.38.05.740.avi",
    #         "2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_01.28.07.709-01.28.11.284.avi",
    #         "2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_02.01.22.526-02.01.26.569.avi",
    #         "0027_The_Big_Lebowski/0027_The_Big_Lebowski_01.29.38.129-01.29.39.478.avi",
    #         "0027_The_Big_Lebowski/0027_The_Big_Lebowski_00.40.50.502-00.40.54.662.avi",
    #         "0027_The_Big_Lebowski/0027_The_Big_Lebowski_01.41.05.074-01.41.10.316.avi",
    #         "0027_The_Big_Lebowski/0027_The_Big_Lebowski_00.29.44.261-00.29.45.192.avi",
    #         "0027_The_Big_Lebowski/0027_The_Big_Lebowski_01.34.35.180-01.34.39.204.avi",
    #         "2012_Unbreakable/2012_Unbreakable_00.42.07.783-00.42.09.797.avi",
    #         "2012_Unbreakable/2012_Unbreakable_00.44.24.867-00.44.27.970.avi",
    #         "2012_Unbreakable/2012_Unbreakable_01.10.54.862-01.10.58.055.avi",
    #         "2012_Unbreakable/2012_Unbreakable_00.37.46.880-00.37.54.563.avi",
    #         "2012_Unbreakable/2012_Unbreakable_00.22.05.930-00.22.21.541.avi",
    #         "1035_The_Adjustment_Bureau/1035_The_Adjustment_Bureau_01.35.24.597-01.35.30.366.avi",
    #         "1035_The_Adjustment_Bureau/1035_The_Adjustment_Bureau_01.16.27.938-01.16.34.205.avi",
    #         "1035_The_Adjustment_Bureau/1035_The_Adjustment_Bureau_01.18.25.654-01.18.29.908.avi",
    #         "1035_The_Adjustment_Bureau/1035_The_Adjustment_Bureau_01.29.55.922-01.29.58.607.avi",
    #         "1035_The_Adjustment_Bureau/1035_The_Adjustment_Bureau_01.17.00.907-01.17.04.533.avi",
    #         "1038_The_Great_Gatsby/1038_The_Great_Gatsby_00.30.55.942-00.30.58.377.avi",
    #         "1038_The_Great_Gatsby/1038_The_Great_Gatsby_00.58.36.915-00.58.39.961.avi",
    #         "1038_The_Great_Gatsby/1038_The_Great_Gatsby_00.55.06.930-00.55.09.734.avi",
    #         "1038_The_Great_Gatsby/1038_The_Great_Gatsby_01.33.48.154-01.33.49.634.avi",
    #         "1038_The_Great_Gatsby/1038_The_Great_Gatsby_01.16.18.342-01.16.20.510.avi",
    #         "0022_Reservoir_Dogs/0022_Reservoir_Dogs_00.09.45.850-00.09.47.403.avi",
    #         "0022_Reservoir_Dogs/0022_Reservoir_Dogs_00.05.27.265-00.05.29.057.avi",
    #         "0022_Reservoir_Dogs/0022_Reservoir_Dogs_01.13.30.094-01.13.33.419.avi",
    #         "0022_Reservoir_Dogs/0022_Reservoir_Dogs_01.19.21.502-01.19.23.673.avi",
    #         "0022_Reservoir_Dogs/0022_Reservoir_Dogs_00.58.40.019-00.58.52.999.avi"]
    scenes = [os.path.join(base_folder, x) for x in os.listdir(base_folder)
                        if x.endswith('avi') or x.endswith('mp4')]

    # Go over all chosen scenes and choose the frames
    save_folder = '/home/hanoch/dataset/msr_vtt/results'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    out_file_csv = open(os.path.join(save_folder, 'result_csv2.csv'), 'w')
    # fieldnames = ['movie name', 'scenes', '1frame', 'fft_center', 'fft_border']
    fieldnames = ['movie name', 'scenes', 'clip_mdf']
    writer = csv.DictWriter(out_file_csv, fieldnames=fieldnames)
    # writer.writerow({'movie name': 'movie name', 'scenes': 'scenes',
    #                  '1frame': '1frame', 'fft_center': 'fft_center', 'fft_border': 'fft_border'})
    writer.writerow({'movie name': 'movie name', 'scenes': 'scenes', 'clip_mdf': 'clip_mdf'})
    scene_detector = NEBULA_SCENE_DETECTOR(use_ClipCap=False, use_OFA=False, use_nebula3=False)
    for scene in scenes:
        # scene here is a short video
        # Detect scenes using the old method (visual change based scene detector)
        scene_elements_old = scene_detector.detect_scene_elements(scene)
        # Go over all the old scenes and detect MDFs using two different methods
        mdfs_fft_center_scenes = ''
        mdfs_fft_border_scenes = ''
        mdfs_1frame_scenes = ''
        clip_mdf = ''
        mdf_det = False # fallback to original

        if mdf_det:
            full_path = os.path.join(base_folder, scene)
            mdfs = scene_detector.detect_mdf(full_path, [scene_elements_old], method='meanshift')
            mdf_outputs = []
            img_caption_outputs = {}
            for mdf in mdfs:
                mdf_images_path = scene_detector.save_mdfs_to_jpg(full_path, mdf, save_path=save_folder)
                if mdf_images_path:
                    print("ka")

        else:
            for idx_scene_element, scene_element in enumerate(scene_elements_old):
                if scene_element[1] - scene_element[0] < 4:
                    continue
                chosen_mdf, ret_img = scene_detector.video_utils.choose_best_frame(os.path.join(base_folder, scene),
                                scene_element[0], scene_element[1])
                clip_mdf = clip_mdf + ' ' + str(chosen_mdf)
                imgname = scene[scene.find('/') + 1:scene.rfind('.')] + f'_old_scene_{chosen_mdf:04}.jpg'
                imgname = os.path.basename(imgname)
                ret = cv2.imwrite(os.path.join(save_folder, imgname), ret_img[0])
                if ret:
                    print("Saved ", os.path.join(save_folder, imgname))

                # mdfs_fft_center_scenes = mdfs_fft_center_scenes + ' ' + create_mdf_string_save_img('fft_center',
                #         scene_element, os.path.join(base_folder, scene), scene_detector, scene, save_folder)
                # mdfs_fft_border_scenes = mdfs_fft_border_scenes + ' ' + create_mdf_string_save_img('fft_border',
                #         scene_element, os.path.join(base_folder, scene), scene_detector, scene, save_folder)
                # mdfs_1frame_scenes = mdfs_1frame_scenes + ' ' + create_mdf_string_save_img('1frame',
                #         scene_element, os.path.join(base_folder, scene), scene_detector, scene, save_folder)

        scene_element_old_str = ''
        for scene_element in scene_elements_old:
            scene_element_old_str = scene_element_old_str + ' ' + str(scene_element[0]) + '-' + str(scene_element[1])

        # writer.writerow({'movie name': scene[scene.find('/') + 1:],
        #                  'scenes': scene_element_old_str,
        #                  '1frame': mdfs_1frame_scenes,
        #                  'fft_center': mdfs_fft_center_scenes,
        #                  'fft_border': mdfs_fft_border_scenes})
        writer.writerow({'movie name': scene[scene.find('/') + 1:],
                         'scenes': scene_element_old_str,
                         'clip_mdf': clip_mdf})

def single_mdf_selection():
    base_folder = '/dataset/lsmdc/avi/'
    scene = '2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_00.06.36.732-00.06.38.382.avi'

    scene_detector = NEBULA_SCENE_DETECTOR(use_ClipCap=False, use_OFA=False, use_nebula3=False)
    scene_elements_clip = scene_detector.detect_scene_elements(os.path.join(base_folder, scene), method='clip')

    for idx_scene_element, scene_element in enumerate(scene_elements_clip):
        # mdfs_old = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element])
        mdfs_new = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element], '1frame')
        mdfs_new = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element], 'fft_center')
        mdfs_new = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element], 'fft_border')
        mdfs_new = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element], 'meanshift')
        print(mdfs_new)

def test_mdf_selection():
    """
    The function test the whole mdf selection pipeline, from scene element detection to actual mdf selection
    :return:
    """

    base_folder = '/dataset/lsmdc/avi/'
    num_movies = 6
    num_scenes = 5
    # Choose randomly num_movies movies and then num_scenes from each movie
    movies = os.listdir(base_folder)
    perm_indexes = np.random.permutation(len(movies))
    chosen_scenes = []
    for k in range(num_movies):
        movie = movies[perm_indexes[k]]
        all_scenes = os.listdir(os.path.join(base_folder, movie))
        scene_perm_indexes = np.random.permutation(len(all_scenes))
        for m in range(num_scenes):
            scene = all_scenes[scene_perm_indexes[m]]
            chosen_scenes.append(os.path.join(movie, scene))

    print('Finished chosing')

    # Go over all chosen scenes and choose the frames
    save_folder = '/home/migakol/data/blur_comp'
    out_file_csv = open(os.path.join(save_folder, 'result_csv.csv'), 'w')
    fieldnames = ['movie name', 'old scenes', 'new scenes', 'old mdfs old scenes', 'new mdfs old scenes',
                  'old mdfs new scenes', 'new mdfs new scenes']
    writer = csv.DictWriter(out_file_csv, fieldnames=fieldnames)
    writer.writerow({'movie name': 'movie name', 'old scenes': 'old scenes', 'new scenes': 'new scenes',
                     'old mdfs old scenes': 'old mdfs old scenes', 'new mdfs old scenes': 'new mdfs old scenes',
                     'old mdfs new scenes': 'old mdfs new scenes', 'new mdfs new scenes': 'new mdfs new scenes'})
    scene_detector = NEBULA_SCENE_DETECTOR(use_ClipCap=False, use_OFA=False, use_nebula3=False)
    for scene in chosen_scenes:
        # scene = '2054_Harry_Potter_and_the_prisoner_of_azkaban/2054_Harry_Potter_and_the_prisoner_of_azkaban_00.06.36.732-00.06.38.382.avi'
        # scene here is a short video
        # Detect scenes using the old method (visual change based scene detector)
        scene_elements_old = scene_detector.detect_scene_elements(os.path.join(base_folder, scene))
        # Detect scenes using CLIP embedding differences
        scene_elements_clip = scene_detector.detect_scene_elements(os.path.join(base_folder, scene), method='clip')
        # Go over all the old scenes and detect MDFs using two different methods
        mdfs_old_old_scenes = ''
        mdfs_new_old_scenes = ''
        for idx_scene_element, scene_element in enumerate(scene_elements_old):
            if scene_element[1] - scene_element[0] < 4:
                continue
            mdfs_old = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element])
            for mdfs in mdfs_old:
                mdfs = list(set(mdfs))
                for mdf in mdfs:
                    mdfs_old_old_scenes = mdfs_old_old_scenes + ' ' + str(mdf)
            mdfs_new = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element], '1frame')
            for mdfs in mdfs_new:
                mdfs = list(set(mdfs))
                for mdf in mdfs:
                    mdfs_new_old_scenes = mdfs_new_old_scenes + ' ' + str(mdf)
            # Store the detected MDFs as images
            old_scenes = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, scene), mdfs_old)
            new_scenes = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, scene), mdfs_new)
            # save the scenes
            k = 0
            for mdf_set in mdfs_old:
                if scene_element[1] in mdf_set:
                    mdf_set.remove(scene_element[1])
                mdf_set = list(set(mdf_set))
                for mdf in mdf_set:
                    filename = scene[scene.find('/')+1:scene.rfind('.')] + f'_old_scene_{mdf:04}.jpg'
                    cv2.imwrite(os.path.join(save_folder, filename), old_scenes[k])
                    k = k + 1

            k = 0
            for mdf_set in mdfs_new:
                if scene_element[1] in mdf_set:
                    mdf_set.remove(scene_element[1])
                mdf_set = list(set(mdf_set))
                for mdf in mdf_set:
                    filename = scene[scene.find('/')+1:scene.rfind('.')] + f'_new_scene_{mdf:04}.jpg'
                    cv2.imwrite(os.path.join(save_folder, filename), new_scenes[k])
                    k = k + 1

        mdfs_old_new_scenes = ''
        mdfs_new_new_scenes = ''
        for idx_scene_element, scene_element in enumerate(scene_elements_clip):
            mdfs_old = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element])
            for mdfs in mdfs_old:
                mdfs = list(set(mdfs))
                for mdf in mdfs:
                    mdfs_old_new_scenes = mdfs_old_new_scenes + ' ' + str(mdf)
            mdfs_new = scene_detector.detect_mdf(os.path.join(base_folder, scene), [scene_element], '1frame')
            for mdfs in mdfs_new:
                mdfs = list(set(mdfs))
                for mdf in mdfs:
                    mdfs_new_new_scenes = mdfs_new_new_scenes + ' ' + str(mdf)
            # Store the detected MDFs as images
            old_scenes = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, scene), mdfs_old)
            new_scenes = scene_detector.video_utils.get_specific_frames(os.path.join(base_folder, scene), mdfs_new)
            # save the scenes
            k = 0
            for mdf_set in mdfs_old:
                if scene_element[1] in mdf_set:
                    mdf_set.remove(scene_element[1])
                mdf_set = list(set(mdf_set))
                for mdf in mdf_set:
                    filename = scene[scene.find('/')+1:scene.rfind('.')] + f'_old_scene_{mdf:04}.jpg'
                    cv2.imwrite(os.path.join(save_folder, filename), old_scenes[k])
                    k = k + 1

            k = 0
            for mdf_set in mdfs_new:
                if scene_element[1] in mdf_set:
                    mdf_set.remove(scene_element[1])
                mdf_set = list(set(mdf_set))
                for mdf in mdf_set:
                    filename = scene[scene.find('/')+1:scene.rfind('.')] + f'_new_scene_{mdf:04}.jpg'
                    cv2.imwrite(os.path.join(save_folder, filename), new_scenes[k])
                    k = k + 1

        scene_element_old_str = ''
        for scene_element in scene_elements_old:
            scene_element_old_str = scene_element_old_str + ' ' + str(scene_element[0]) + '-' + str(scene_element[1])

        scene_elements_clip_str = ''
        for scene_element in scene_elements_clip:
            scene_elements_clip_str = scene_elements_clip_str + ' ' + str(scene_element[0]) + '-' + str(scene_element[1])

        writer.writerow({'movie name': scene[scene.find('/')+1:],
                         'old scenes': scene_element_old_str,
                         'new scenes': scene_elements_clip_str,
                         'old mdfs old scenes': mdfs_old_old_scenes,
                         'new mdfs old scenes': mdfs_new_old_scenes,
                         'old mdfs new scenes': mdfs_old_new_scenes,
                         'new mdfs new scenes': mdfs_new_new_scenes})


def main():
    scene_detector = SceneDetector()
    # scene_detector.update_s2_clsmdc_with_indices()
    # scene_detector.update_s1_lsmdc_with_ofa()
    concate_movies = False
    # We concatenate 20 movies of LSMDC
    if concate_movies:
        lsmdc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset1/lsmdc/")
        scene_detector.create_concat_lsmdc_movies(lsmdc_dir=lsmdc_path, num_of_movies=200, base_path = "/dataset1/", dest_path = "/dataset1/concatenated_mp4_200")

    # path = '/dataset/development/1010_TITANIC_00_41_32_072-00_41_40_196.mp4'
    # scene_detector.divide_movie_by_timestamp(path, 3, 4)

    # scene_detector.new_movies_batch_processing()

    #scene_detector.init_new_db("nebula_datadriven")

    # INSERT concatenated movies to database
    good_videos_lst = [
    '0026_The_Big_Fish_01.27.55.748-01.28.12.028.mp4',
    '3006_A_GOOD_DAY_TO_DIE_HARD_01.30.26.150-01.30.41.504.mp4',
    '3019_COLOMBIANA_00.07.07.000-00.07.21.393.mp4',
    '3042_KARATE_KID_01.19.19.995-01.19.34.148.mp4',
    '3036_IN_TIME_01.22.02.304-01.22.19.184.mp4',
    '3042_KARATE_KID_01.59.59.749-02.00.13.205.mp4',
    '1061_Harry_Potter_and_the_deathly_hallows_Disk_Two_01.02.51.000-01.03.10.020.mp4',
    '3017_CHRONICLE_01.07.20.988-01.07.33.210.mp4',
    '3036_IN_TIME_00.28.05.713-00.28.23.353.mp4',
    '1018_Body_Of_Lies_01.30.12.330-01.30.28.851.mp4',
    '3043_KATY_PERRY_PART_OF_ME_00.52.01.170-00.52.20.423.mp4',
    '3033_HUGO_01.31.07.290-01.31.23.468.mp4',
    '3089_XMEN_FIRST_CLASS_01.54.31.621-01.54.51.029.mp4',
    '1046_Australia_01.03.48.582-01.04.07.197.mp4',
    '3006_A_GOOD_DAY_TO_DIE_HARD_01.08.46.435-01.09.06.404.mp4',
    '3049_MORNING_GLORY_01.40.02.750-01.40.17.024.mp4',
    '0028_The_Crying_Game_00.36.14.572-00.36.29.917.mp4',
    '3004_500_DAYS_OF_SUMMER_00.13.21.834-00.13.35.933.mp4',
    '1047_Defiance_00.30.41.989-00.31.03.954.mp4',
    '3013_BURLESQUE_01.45.48.267-01.46.05.903.mp4'
    ]
    mp4_path = "/dataset1/concatenated_mp4_200/"
    _files = [f for f in os.listdir(mp4_path) if f.endswith(".mp4")]
    prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static"
    source = "lsmdc_concatenated"
    #Example usage
    for idx, _file in enumerate(_files):
        file_name = basename(_file)
        movie_name = file_name.split(".mp4")[0]
        url_link = os.path.join(prefix_link+mp4_path, movie_name + ".mp4")
        full_path = os.path.join(mp4_path, movie_name + ".mp4")
        # print(f"number: {idx+1}")
        # print(f"url_link: {url_link}")
        # print("Conclusion: ")
        # print("-"*30)
        if movie_name + ".mp4" in good_videos_lst:
            # scene_detector.insert_movies_pipeline(full_path, url_link, source, db_name="ilan_test")
            scene_detector.insert_movie(full_path, url_link, source, db_name="prodemo")

import clip
def encode_text(clip_util, text):
    text_token = torch.cat([clip.tokenize(text)]).to('cpu')
    text_emb = clip_util.model.encode_text(text_token).detach().numpy()
    text_emb = text_emb / np.linalg.norm(text_emb)
    return text_emb

def compare_text(clip_util, img_emb, text):
    text_emb = encode_text(clip_util, text)
    return np.sum(text_emb * img_emb)

if __name__ == "__main__":
    paperspace_playground_mk()

