from realmparser import VodParse, RealmParser
import pandas as pd
import json
import cv2 as cv
from time import time
from glob import glob

the_role = "mid"
video_name = "mid"
test_region = "map"
img_paths = glob("realmparser/resolutions/*.png")

mid_realm = RealmParser(
    realm_path=img_paths[0],
    input_region="map",
    lol_cfg_path="game.cfg",
    role=the_role,
)
for img_path in img_paths:
    img = cv.imread(img_path)
    test_crop = VodParse.get_mask(mid_realm, img)
    resize_crop = VodParse.resize_by_pxpercent(
        mid_realm, test_crop, desired_px=1000, w_or_h="h"
    )
    cv.imshow("Testing", resize_crop)
    print(img_path)
    cv.waitKey(1000)
