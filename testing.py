import json
from glob import glob
from pprint import pprint
from time import time

import cv2 as cv
import numpy as np
import pandas as pd
import plotly.express as px
from skimage import io
from realmparser import RealmParser, VodParse

the_role = "mid"
video_name = "mid"
test_region = "map"
img_paths = glob("realmparser/resolutions/*.png")

mid_realm = RealmParser(
    realm_path=f"{the_role}.mp4",
    input_region=test_region,
    lol_cfg_path="game.cfg",
    role=the_role,
)


def compile_realm_video(read_imgs: list, compile_type: str):
    compilation_cfg = {
        "scoreboard": {
            "height": 30,
            "width": 1000,
        },
        "map": {
            "height": 1000,
            "width": 1000,
        },
    }
    img_qty = len(read_imgs)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(
        f"{compile_type}.avi",
        fourcc,
        1,
        (
            compilation_cfg[compile_type]["height"],
            compilation_cfg[compile_type]["width"],
        ),
    )

    for frame_number, img in enumerate(read_imgs):
        video.write(img)
        # cv.imshow("Test", img)
        # cv.waitKey(1000)

    cv.destroyAllWindows()
    video.release()

    # Writes the the output image sequences in a video file


def plotly_img_timeline(imgs: list):
    fig = px.imshow(
        np.array(imgs),
        animation_frame=0,
        binary_string=True,
    )
    fig.show()


### Map Crop Testing
## Note for Reference: shape[0]=height; shape[1]=width
# get imgs
my_imgs = [cv.imread(img_path) for img_path in img_paths]
img_heights = np.array([int(img_path.split(".")[1]) for img_path in img_paths])
img_widths = np.array(
    [
        int(img_path.split(".")[0].replace("realmparser/resolutions\\", ""))
        for img_path in img_paths
    ]
)

# mask imgs... should be square img, i.e. pprint should show array of 0's
masked_imgs = [VodParse.get_mask(mid_realm, img) for img in my_imgs]
masked_heights = np.array([int(img.shape[0]) for img in masked_imgs])
masked_widths = np.array([int(img.shape[1]) for img in masked_imgs])
# array of 0's means that we got a square map-- good
pprint(masked_heights - masked_widths)

# if masked imgs are squares, there should be no issues resizing
resized_imgs = [
    VodParse.resize_by_pxpercent(mid_realm, img, desired_px=1000, scale_wrt_w_or_h="h")
    for img in masked_imgs
]
resized_heights = np.array([resized_img.shape[0] for resized_img in resized_imgs])
resized_widths = np.array([resized_img.shape[1] for resized_img in resized_imgs])
# array of 0's means that we got a square map-- good
pprint(resized_heights - resized_widths)

compile_realm_video(resized_imgs, "map")

plotly_img_timeline(resized_imgs)
# for img_index, img in enumerate(resized_imgs):
#     fig = px.imshow(img)
#     fig.show()

print("End. Placeholder for debug pause.")

### Crop to player portrait
