from fastapi import FastAPI
from PIL import Image
from typing import List
from joblib import parallel_backend
from matplotlib.cbook import flatten
from pydantic import BaseModel
import validators
import os
import uuid
import sys
import numpy as np
from experts.common.constants import OUTPUT_DB, OUTPUT_DB_JSON, TYPE_IMAGE, TYPE_MOVIE
from experts.service.base_expert import BaseExpert, DEFAULT_FILE_PATH
from experts.app import ExpertApp
from experts.common.models import ExpertParam, TokenRecord, ImageRecord
from experts.common.defines import OutputStyle
import experts.common.constants as constants
from paramiko import SSHClient, AutoAddPolicy
import paramiko
# from contextlib import closing
from scp import SCPClient
# from scpclient import WriteDir, closing
import shutil
import datetime
import time
import requests
#### for running outside the uvicorn
# sys.path.append('/notebooks')
# sys.path.append('/home/hanoch/notebooks/nebula3_videoprocessing')
# sys.path.append('/notebooks/nebula3_videoprocessing')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

#####
from expert.commons.config import VideoProcessingConf
from expert.scene_detect import SceneDetector
from scenedetect import VideoManager
from database.utils import make_hash
from database.arangodb import DatabaseConnector, DBBase
from experts.pipeline.api import *
from experts.pipeline.api import PipelineConf


URL_TYPE_FILE = 'file' # in contrast to URL
URL_TYPE_STREAM = 'stream'
MOVIES_COLLECTION_NAME = 'Movies'
RESULT = 'result'
RESULTS = 'results'
INPUT_TYPE_DATASET = 'dataset'
PATH = 'path'
FILE_PREFIX = 'file://'
FILE_PREFIX_LEN = len(FILE_PREFIX)
CONTEXT = 'context'

os.environ["CONFIG_NAME"] = "default" 

avoid_duplicates = True

class MovieParam(BaseModel):
    movie_id: str = None
    url: str
    type: str = URL_TYPE_FILE

class ProcessMoviesParam(BaseModel):
    is_async: bool = False,
    movies: List[MovieParam] = None
    dataset: dict = None,
    context: dict = None,
    output: str = constants.OUTPUT_JSON
    input: str = None
    overwrite: bool = False
    save_movies: bool = False,
    benchmark: dict = None

    class Config:
        schema_extra = {
            "example": {
                "image_id": "The image id",
                "url": "url to the image (remote: 'http://..' , local: 'file://...')",
                "save_movies": "save movies and frames to disk",
                "output": "where to output: json (return json in response)/db/db+json, default- json",
                "is_async": "will this api runs async, default: False",
                "dataset": "dataset parans (dict)",
                "context": "dict params of context",
            }
        }

class ProcessMovieParam(BaseModel):
    movie_id: str
    url: str
    movie_suffix: str = None
    save_movie: bool = False
    output: str = constants.OUTPUT_JSON
    overwrite: bool = False
    context: str = None
    type: str = constants.TYPE_MOVIE,
    dataset_id: str = "",
    benchmark_name: str = "",
    benchmark_tag: str = ""


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

class VideoProcessingExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        self.vp_config = VideoProcessingConf()
        self.scene_detect = SceneDetector()
        self.nebula_db = DBBase()        
        # make dirs
        if not os.path.isdir(self.vp_config.get_local_frames_path()):
            os.mkdir(self.vp_config.get_local_frames_path())
        # init ssh
        userpass = self.vp_config.get_web_userpass().split(":")
        self.ssh = SSHClient()
        self.ssh.set_missing_host_key_policy(AutoAddPolicy())
        # self.ssh.load_system_host_keys()
        self.ssh.connect(self.vp_config.get_web_host(),
                         username=userpass[0],password=userpass[1])
        self.scp = SCPClient(self.ssh.get_transport())

        # self.ssh_client = paramiko.SSHClient()
        # self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # self.ssh_client.connect(self.vp_config.get_web_host(),
        #                  username=userpass[0],password=userpass[1])
        # after init all
        self.set_active()
        self.settings = PipelineConf()
        self.video_proc_settings = self.settings["video_processing"]
        self.video_proc_settings_param = dict()
        if self.video_proc_settings == "every_n_seconds":
            self.video_proc_settings_param["every_n_seconds"] = self.settings["video_processing_interval"]

        

    def set_app(self, app: ExpertApp):
        self.exprt_app = app

    def get_name(self):
        return "videoprocessing"

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        return { 'error': 'not implemented' }

    def predict_image(self, expert_params: ExpertParam):
        """ handle new image """
        return { 'error': 'not implemented' }

    def validate_process_movies_params(self, params: ProcessMoviesParam):
        print("Params: {}".format(params))
        if params.input and params.input == INPUT_TYPE_DATASET and params.dataset:
            return self.get_dataset_movies(params)
        return self.get_url_movies(params)

    def get_url_movies(self, params: ProcessMoviesParam):
        print("URL Params: {}".format(params))
        movie_list = []
        config_tag = self.get_config_tag(os.getenv("PIPELINE_ID"))
        if len(params.movies) > 0:
            for movie in params.movies:
                if not validators.url(movie.url):
                    print(f'url not valid: {movie.url}, skipping')
                    continue
                if not movie.movie_id or len(movie.movie_id) == 0:
                    # set movie id
                    movie_name = movie.url.rsplit("/",1)[-1]
                    movie.movie_id = movie_name.rsplit(".",1)[0]
                movie_suffix = movie.url.rsplit(".",1)[1]
                movie_param = ProcessMovieParam(movie_id=movie.movie_id,
                                                url=movie.url,
                                                movie_suffix=movie_suffix,
                                                save_movie=params.save_movies,
                                                output=params.output,
                                                overwrite=params.overwrite,
                                                type=movie.type,
                                                dataset_id="",
                                                benchmark_name=params.benchmark['benchmark_name'] if params.benchmark else "",
                                                benchmark_tag=params.benchmark['benchmark_tag'] if params.benchmark else config_tag)
                movie_list.append(movie_param)
        else:
            return movie_list, 'No movies in the list'
        return movie_list, None

    def get_dataset_movies(self, params: ProcessMoviesParam):
        print("Dataset parameters: {}".format(params))
        movie_list = []
        movie_context = None
        context_delim = params.context["delimeter"] if params.context and params.context["delimeter"] else None
        if params.dataset[PATH] is None:
            return movie_list, "no dataset path"

        dataset_path = params.dataset[PATH]
        movie_files = os.listdir(dataset_path)
        config_tag = self.get_config_tag(os.getenv("PIPELINE_ID"))
        if len(movie_files) > 0:
            for movie_file in movie_files:
                if not os.path.isfile(os.path.join(dataset_path, movie_file)):
                    continue
                if context_delim:
                    movie_parts = movie_file.split(context_delim)
                    if len(movie_parts) >= 2:
                        movie_context = movie_parts[0]
                    # set movie id
                    movie_id = movie_file.rsplit(".",1)[0]
                    movie_suffix = movie_file.rsplit(".",1)[1]
                    url = FILE_PREFIX + os.path.join(dataset_path, movie_file)
                    movie_param = ProcessMovieParam(movie_id=movie_id,
                                                    url=url,
                                                    movie_suffix=movie_suffix,
                                                    save_movie=params.save_movies,
                                                    output=params.output,
                                                    overwrite=params.overwrite,
                                                    context=movie_context,
                                                    type=params.dataset['type'],
                                                    dataset_id=params.dataset['id'],
                                                    benchmark_name=params.benchmark['benchmark_name'] if params.benchmark else "",
                                                    benchmark_tag=params.benchmark['benchmark_tag'] if params.benchmark else config_tag)
                    movie_list.append(movie_param)
        else:
            return movie_list, 'No movies in the list'
        return movie_list, None

    def add_expert_apis(self, app: FastAPI):
        @app.post("/process/movies")
        def post_process_movies(params: ProcessMoviesParam):
            # validate
            process_movies_list, error = self.validate_process_movies_params(params)
            if error:
                return { RESULT: None, 'error': error}

            job_id = str(uuid.uuid4())
            with self.exprt_app.jobs_lock:
                self.exprt_app.job_cache[job_id] = { "state": "running", "params" :process_movies_list }
            future = self.exprt_app.pool.submit(self.handle_process_movies, process_movies_list, job_id)
            if params.is_async:
                return { 'jobId': job_id}
            else:
                response = future.result(self.exprt_app.config.get_max_wait_predict_time()) # self.expert.predict_image(expert_params)
                return response

    def handle_process_movies(self, process_movies_list: List[ProcessMovieParam], job_id: str, is_task: bool = False):
        results = list()
        for process_movie_param in process_movies_list:
            try:
                results.append(self.process_movie(process_movie_param))
            except Exception as e:
                results.append({ RESULT: None, 'error': f'failed processing movie: {process_movie_param.url} with exception: {str(e)}' })
        if not is_task:
            with self.exprt_app.jobs_lock:
                self.exprt_app.job_cache[job_id] = { "state": "finish", "results": results }
        return { 'results' : results,  'jobId': job_id }

    def upload_files_to_web(self, uploads):
        try:
            # for upload in uploads:
            for local_path, remote_path in uploads.items():
                self.scp.put(local_path, recursive=True, remote_path=remote_path)
        except Exception as exp:
            print(f'An exception occurred: {exp}')
            return False
        return True

    def process_movie(self, process_movie_param: ProcessMovieParam):
        # download movie to location if not exists and not overwrite
        print("Process movie params: {}".format(process_movie_param))
        error = None
        movie_file = None
        result = {}
        if process_movie_param.url.startswith(FILE_PREFIX):
            movie_file = process_movie_param.url[FILE_PREFIX_LEN:]
            tmp_movie_path = f'{self.vp_config.get_local_movies_path()}/{process_movie_param.movie_id}.{process_movie_param.movie_suffix}'
            shutil.copy2(movie_file, tmp_movie_path)
            movie_file = tmp_movie_path
            # movie_file = process_movie_param.url[FILE_PREFIX_LEN:]
            # just check for avi suffix
            if process_movie_param.movie_suffix == "avi":
                # conversion should be done on a local writeble folder
                self.scene_detect.convert_avi_to_mp4(movie_file, movie_file.replace(".avi",""))
                movie_file = movie_file.replace(".avi",".mp4")
        else:
            movie_file, error = self.download_movie(process_movie_param)

        orig_url = process_movie_param.url
        movie_url_path = f'{movie_file.replace(self.vp_config.get_local_movies_path(),self.vp_config.get_movies_path())}'#self.vp_config.get_movies_path()
        #f'{movie_file.replace(self.vp_config.get_local_movies_path(),self.vp_config.get_movies_path())}'
        mdfs_local_dir = f'{self.vp_config.get_local_frames_path()}/{process_movie_param.movie_id}'
        mdfs_web_dir = f'{self.vp_config.get_frames_path()}' #/{process_movie_param.movie_id}'
        scenes = []
        mdfs = []
        mdfs_path = []
        scene_elements = []
        if process_movie_param.type == constants.TYPE_MOVIE: # for images just upload and write to DB
            # getting scene elements
            print("Processing video...")
            scenes = self.scene_detect.detect_scenes(movie_file)
            scene_elements = self.scene_detect.detect_scene_elements(movie_file)
            if 'every_n_seconds' in self.video_proc_settings_param:
                mdfs = self.scene_detect.get_mdf_every_n_sec(movie_file, n_sec=self.video_proc_settings_param['every_n_seconds'])
            else:
                mdfs = self.scene_detect.detect_mdf(movie_file, scene_elements, method='meanshift')
            movie_meta = self.get_movie_meta(movie_file)
            # create
            # saving mdfs on disk

            mdfs_path = []
            if len(mdfs):
                mdfs_path = self.save_mdfs_frames(process_movie_param, mdfs, movie_file)
                if len(mdfs_path) == 0:
                    return { RESULT: None, 'error': f'failed to save mfds of movie url: {process_movie_param.url}'}
                # convert mdfs_path to remote
                mdfs_path = [mdf.replace(self.vp_config.get_local_frames_path(),self.vp_config.get_frames_path()) for mdf in mdfs_path]
                print("mdfs_path: {}".format(mdfs_path))
            # upload movie file and mdfs
            # mdfs_web_path = [{
            #     mdf: f'{self.vp_config.get_web_prefix()}/{mdf.replace(self.vp_config.get_local_movies_path(),self.vp_config.get_movies_path())}'
            #     } for mdf in mdfs_path]
            # uploads = {
            #     movie_file: movie_url_path ,
            #     mdfs_local_dir: mdfs_web_dir
            # }
            if "youtube" in orig_url:
                movie_url_path_yt = '/'.join(movie_url_path.split("/")[:-2])
                movie_file_yt = '/'.join(movie_file.split("/")[:-1])
                uploads = {
                    movie_file_yt: movie_url_path_yt,
                    mdfs_local_dir: mdfs_web_dir
                }
            else:
                uploads = {
                    movie_file: movie_url_path,
                    mdfs_local_dir: mdfs_web_dir
                }
            # folder_upload = {mdfs_local_dir: mdfs_web_dir}
            # mdf_names = {os.path.join(mdfs_local_dir, mdf_path.split("/")[-1]) : mdfs_web_dir for mdf_path in mdfs_path}
            # uploads.append(mdf_names)
        elif process_movie_param.type == constants.TYPE_IMAGE:
            movie_meta = self.get_image_meta(movie_file)
            mdfs_path = [movie_file.replace(self.vp_config.get_local_movies_path(),self.vp_config.get_movies_path())]
            uploads = {
                movie_file: movie_url_path,
            }
            mdfs = [[0]]
        else:
            raise
        # mdf_names = {os.path.join(mdfs_local_dir, mdf_path.split("/")[-1]) : mdfs_web_dir for mdf_path in mdfs_path}
        print("Uploads: {}".format(uploads))
        print("MDFs: {}".format(mdfs_path))
        # self.upload_files_to_web(uploads, folder_upload)
        self.upload_files_to_web(uploads)
        # insert movie info to DB
        
        # dataset_name = ""
        # if process_movie_param.dataset_id:
        #     dataset_name = process_movie_param.dataset_id.split("_batch")[0]
        config_name = self.get_config_name(os.getenv("PIPELINE_ID"))
        config_tag = self.get_config_tag(os.getenv("PIPELINE_ID"))
        movie_entry = {
            "name": process_movie_param.movie_id,
            "url_path": movie_url_path, #"http://74.82.28.99:9000/msrvtt/video8351.mp4",
            "orig_url": orig_url,
            "scenes": scenes if len(scenes) else [], # [[0,375]],
            "scene_elements": scene_elements if len(scene_elements) else [],
            "mdfs": mdfs if len(mdfs) else [],
            "mdfs_path": mdfs_path if len(mdfs_path) else [],
            "meta": movie_meta,
            CONTEXT: process_movie_param.context if process_movie_param.context else "",
            "updates": 1,
            "source": "external",
            # "dataset_name": dataset_name,
            "pipeline_id": os.getenv("PIPELINE_ID", None),
            "misc": {
                        "benchmark_name": process_movie_param.benchmark_name if process_movie_param.benchmark_name else "",
                        "config_name": config_name,
                        "benchmark_tag": process_movie_param.benchmark_tag if process_movie_param.benchmark_tag else config_tag,
                        "execution_time": '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() )
                    }   
        }
        # save in db
        result = self.save_do_db(process_movie_param, movie_entry)
        print(result)
        print("---------")
        result[CONTEXT] = process_movie_param.context if process_movie_param.context else ""
        print(result)
        print("--------")
        # return
        return { RESULT: result, 'error': error }

    def get_config_name(self, pipeline_id):
        rc = self.nebula_db.get_doc_by_key({'_key': pipeline_id}, 'pipelines')
        if "config_name" in rc:
            return rc['config_name']
        return ""

    def get_config_tag(self, pipeline_id):
        rc = self.nebula_db.get_doc_by_key({'_key': pipeline_id}, 'pipelines')
        if "tag" in rc:
            return rc['tag']
        return ""

    def save_do_db(self, process_movie_param: ProcessMovieParam, movie_entry):
        result = None
#         collection = self.movie_db.db.collection(name = MOVIES_COLLECTION_NAME)
        # doc = { '_key':  f'{token.id}:{token.expert}:{token.scene}:{token.label.replace(" ","_")}', **token.__dict__ }
        doc = movie_entry # { **movie_entry.__dict__ }
        if avoid_duplicates:
            doc['_key'] = str(make_hash(process_movie_param.url))
        result, _ = self.nebula_db.write_doc_by_key(doc,MOVIES_COLLECTION_NAME,overwrite=avoid_duplicates,key_list=['_key'])
        print(doc)
        print("---------")        
        return result

    def download_movie(self, process_movie_param: ProcessMovieParam):
        ''' create a movie file path and download to it if is doesn't exist and no overwrite
            returns the movie_file path '''
        error = None

        # movie_name.split('.')[:-1][0] + '_' + str(int(time.time())) + '.' + movie_name.split('.')[-1:][0]       
        
        # movie_name = str(int(time.time())) + '_' + movie_name
        movie_name = f'{process_movie_param.movie_id}.{process_movie_param.movie_suffix}'
        movie_file = f'{self.vp_config.get_local_movies_path()}/{movie_name}'

        if "youtube" in process_movie_param.url:
            movie_name = f'{process_movie_param.movie_id}'.split('=')[-1]
            movie_file = f'{self.vp_config.get_local_movies_path()}/{movie_name}'
        # if the file exists and not overwrite then exit
        if os.path.exists(movie_file) and not process_movie_param.overwrite:
            error = f'movie already exists: {movie_file}'
            return movie_file, error
# Modify movie_name
        # movie_name = movie_name + '_' + str(int(time.time()))
        # altough it's named image, it is file in general
        movie_file, error =  self.download_image_file(image_url=process_movie_param.url,
                                         image_location=movie_file,
                                         remove_prev=process_movie_param.overwrite)
        if "youtube" in process_movie_param.url:
            process_movie_param.movie_suffix = "mp4"
            process_movie_param.movie_id = movie_name
            process_movie_param.type = constants.TYPE_MOVIE
        # check if we need to convert to mp4
        if process_movie_param.movie_suffix == "avi":
            self.scene_detect.convert_avi_to_mp4(movie_file, movie_file.replace(".avi",""))
            movie_file = movie_file.replace(".avi",".mp4")
        if process_movie_param.type == constants.TYPE_IMAGE:
            if process_movie_param.movie_suffix.lower() == "png" or \
                process_movie_param.movie_suffix.lower() == "jpg" or \
                process_movie_param.movie_suffix.lower() == "jpeg":
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', \
                                 "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", \
                                 "Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
                    resp = requests.get(process_movie_param.url, stream=True, headers=headers).raw
                    if not resp:
                        f'Failed to download movie: {process_movie_param.url}'
                    im1 = Image.open(resp)#.convert('RGB')
                    # im1 = Image.open(movie_file)
                    movie_file = movie_file.replace("%", "")
                    im1.save(movie_file)
            else:
                 error = f'Only JPEG, JPG and PNG are supported image formats, you tried this format: {process_movie_param.movie_suffix}'
                 raise ValueError(error)
        return movie_file, error

    def get_movie_meta(self, movie_file):
        video_manager = VideoManager([movie_file])
        return {
            'fps': video_manager.frame_rate,
            'width': video_manager.frame_size[0],
            'height': video_manager.frame_size[1],
        }

    def get_image_meta(self, image_file):
        im1 = Image.open(image_file)
        return {
            'mode': im1.mode,
            'width': im1.width,
            'height': im1.height,
        }

    def save_mdfs_frames(self, process_movie_param: ProcessMovieParam, mdfs, movie_file):
        mdfs_path = []
        frame_list = tuple(sum(mdfs, []))
        frames_location = f'{self.vp_config.get_local_frames_path()}/{process_movie_param.movie_id}'
        ret_frames = self.divide_movie_into_frames(frame_list=frame_list,
                                                   movie_location=movie_file,
                                                   movie_out_folder=frames_location,
                                                   remove_prev=process_movie_param.overwrite)
        return ret_frames

    def run_batch(self):
        """handle batch mode
        """
        print(f'running batch from: {self.get_name()}')

    def run_pipeline_task(self):
        """ build the movie list in the movies_status
            then update the pipeline and start the pipeline task
            since this expert creates the movie entries, to avoid
            dupilcate work we don't use the PipelineTask that process
            we use process movies directly
        """
        pipeline_id = os.getenv("PIPELINE_ID",None)
        if not pipeline_id:
            print(f'failed to get pipeline_id in: {self.get_name()}')
            return

        print(f'running pipeline task from: {self.get_name()}, pipeline-id: {pipeline_id}') 
        # for debug change db : self.nebula_db.change_db('web_demo')  pipeline.movie_db.change_db('web_demo') pipeline.config.ARANGO_DB = 'web_demo'
        pipeline = PipelineApi(None)
        pipeline_doc = pipeline.get_pipeline(pipeline_id)
        print(pipeline_doc)
        print("------")
        # now update start work of videoprocessing from the pipeline doc
        if not pipeline_doc:
            print(f'failed to get pipeline for {self.get_name()}, exit...')
            return
        # check if the expert already run
        if  pipeline_doc[PIPELINE_TASKS] and \
            pipeline_doc[PIPELINE_TASKS][self.get_name()] and \
            pipeline_doc[PIPELINE_TASKS][self.get_name()] == STATUS_SUCCESS:
            print(f'pipeline: {pipeline_id} already processed successfuly by {self.get_name()}, exit...')
            return

        # start work on movies
        # build Params
        vp_input = pipeline_doc[PIPELINE_INPUTS][self.get_name()]
        process_params = ProcessMoviesParam(**vp_input)
        print("process_params: {}".format(process_params))
        process_movies_list, error = self.validate_process_movies_params(process_params)
        if error:
            print(f'failed to validate input params for pipeline: {pipeline_id}, expert: {self.get_name()}, exit...')
            return
        self.pipeline_params = process_params

        # handle movie list
        movies_status = {}
        process_status = self.handle_process_movies(process_movies_list, pipeline_id, True)
        results = process_status[RESULTS]
        print("-------")
        print(results)
        if results and len(results):
            # create movie list and update pipeline
            for entry in results:
                result = entry[RESULT]
                if result:
                    movies_status[result['_id']] = {
                        '_key': result['_key'],
                        CONTEXT: result[CONTEXT],
                        'status': { self.get_name(): STATUS_SUCCESS }
                    }
            if len(movies_status): # write to pipeline
                pipeline.update_pipeline(task_name=self.get_name(),
                                         pipeline_id=pipeline_id,
                                         movies_status=movies_status,
                                         task_status={self.get_name(): STATUS_SUCCESS },
                                         task_inputs={})


my_expert = VideoProcessingExpert()
expert_app = ExpertApp(expert=my_expert)
app = expert_app.get_app()
my_expert.set_app(expert_app)
expert_app.run()
