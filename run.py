import realmparser
import pandas as pd
import json
import cv2 as cv
from time import time

video_name = "CHANGE ME"
mid_realm = realmparser.VodParse(
    realm_path=f"{video_name}.mp4",
    input_region="scoreboard",
    lol_cfg_path="game.cfg",
    role="CHANGE ME",
    des_time_interval=5,
)

df = pd.DataFrame.from_dict(mid_realm.output)
df.to_csv("clean_data.csv")


# df.to_pickle("easyocr_testpkl.pkl")
# # with open(f"{video_name}.json", "w") as outfile:
# #     json.dump(myvod.output_dict, outfile)
# df.to_excel(f"cleandata.xlsx", sheet_name="RealmParse")
