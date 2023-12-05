from os import getenv

"""backend & models
VideoPredictor init params:
- tracker backend: detectron or tflow
- pretrained_model_cfg - default: CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x
- confidence_threshold - default 0.5

"""

class VideoProcessingConf:
    def __init__(self) -> None:

        self.MOVIES_PATH = getenv('MOVIES_PATH', '/datasets/media/movies')
        self.LOCAL_MOVIES_PATH = getenv('LOCAL_MOVIES_PATH', '/tmp')
        self.FRAMES_PATH = getenv('FRAMES_PATH','/datasets/media/frames')
        self.LOCAL_FRAMES_PATH = getenv('LOCAL_FRAMES_PATH','/tmp/frames')
        self.WEB_PREFIX = getenv('WEB_PREFIX', 'http://74.82.29.209:9000')
        self.WEB_HOST = getenv('WEB_HOST', '74.82.29.209')
        self.WEB_USERPASS = getenv('WEB_USERPASS', 'paperspace:Nebula@12345')

    def get_movies_path(self):
        return (self.MOVIES_PATH)
    def get_local_movies_path(self):
        return (self.LOCAL_MOVIES_PATH)
    def get_frames_path(self):
        return (self.FRAMES_PATH)
    def get_local_frames_path(self):
        return (self.LOCAL_FRAMES_PATH)
    def get_web_prefix(self):
        return (self.WEB_PREFIX)
    def get_web_host(self):
        return (self.WEB_HOST)
    def get_web_userpass(self):
        return (self.WEB_USERPASS)
