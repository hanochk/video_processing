# nebula3_videoprocessing
Tools for videoprocessing of NEBULA3.

# How to use:
a. `git clone https://github.com/NEBULA3PR0JECT/nebula3_videoprocessing.git`

b. `cd nebula3_videoprocessing`

c. `conda env create --name videoprocessing --file=environment.yml`

d. `conda activate videoprocessing`

# in case CLIP is needed
pip install git+https://github.com/openai/CLIP.git

# What this repo includes (currently):

1. `scene_detector_api.py` which is responsible for scene detection, detecting mdfs, conversion of avi to mp4, detecting scene elements.

2. `s3_frame_upload_api.py` which is responsible for storing frames in db, storing frame to s3, inserting node to scenegraph and dividing movie into frames. 

