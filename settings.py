import os

import numpy as np

DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "runwayml/stable-diffusion-v1-5")

MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "4"))
DEFAULT_NUM_IMAGES = min(MAX_NUM_IMAGES, int(os.getenv("DEFAULT_NUM_IMAGES", "2")))
MAX_IMAGE_RESOLUTION = int(os.getenv("MAX_IMAGE_RESOLUTION", "768"))
DEFAULT_IMAGE_RESOLUTION = min(MAX_IMAGE_RESOLUTION, int(os.getenv("DEFAULT_IMAGE_RESOLUTION", "512")))

ALLOW_CHANGING_BASE_MODEL = os.getenv("SPACE_ID") != "hysts/ControlNet-v1-1"
SHOW_DUPLICATE_BUTTON = os.getenv("SHOW_DUPLICATE_BUTTON") == "1"

MAX_SEED = np.iinfo(np.int32).max
