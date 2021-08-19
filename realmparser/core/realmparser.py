import configparser
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from datetime import datetime
from glob import glob
from pprint import pprint
from time import time

import cv2 as cv
import easyocr
import numpy as np
import pandas as pd

START_TIME = time()


def _path_exists(file_path: str):
    """Checks if file exists at the entered path.

    Args:
        file_path (str): Path to check.

    Returns:
        Bool: True if path exists, false if not.
    """
    if glob(file_path):
        # then file exists
        return True
    elif glob is []:
        return FileNotFoundError(f"No file found at path: {file_path}")


def intpercent(myinteger: int, percent: float, add_px=0):
    """Returns a percent of an integer as an integer.

    Args:
        myinteger (int): Integer to take percent of.
        percent (float): Percent to return.
        add_px (int): Add scalar amount to final value.

    Returns:
        output (int): Final value
    """
    output = round((myinteger * percent), 2)
    output = int(output + add_px)

    return output


class RealmParser:
    def __init__(
        self,
        realm_path: str,
        input_region: str,
        lol_cfg_path: str,
        role: str,
    ):
        """Parse a LoL video or image of gameplay into a dictionary.

        Args:
            realm_path (str): Path to the video or img to be OCR'ed
            input_region (str): Region to parse. Accepted Args: "scoreboard",
            lol_cfg_path (str): Path to player's config file.
            role (str): Any descriptive string. Design intent is to allow unique id of RealmParse objs
        """

        # Initialize vars that are mostly independant
        print("Starting RealmParse...")

        self.input_region = input_region
        self.role = role

        self.TYPE_SWITCH = {
            np.ndarray: "self.ImgParse()",
            cv.VideoCapture: "self.VodParse()",
        }
        self.lol_cfg = {}
        self.ocr_read_list = []
        self.output = []

        # configuration containing most info needed to crop to that region
        self.region_cfg = {
            "scoreboard": {
                "mask_coords": {
                    "y1": "0",
                    "y2": "intpercent(height, 0.0303, 1)",
                    "x1": "int(width - intpercent(height, 0.396, 1))",
                    "x2": "int(width)",
                },
                "ocr_whitelist": "0123456789:/",
            },
            "map": {
                "mask_coords": {
                    "y1": "intpercent(height,0.74125)",
                    "y2": "int(height)",
                    "x1": "int(width - intpercent(height,0.2587,1))",
                    "x2": "int(width)",
                },
                "mirror_mask_coords": {},
            },
            "player_hud": {
                "mask_coords": {
                    "y1": "",
                    "y2": "",
                    "x1": "",
                    "x2": "",
                },
            },
        }
        # configuration containing the settings for cropping & getting best ocr reads from each subregion in a region
        # TODO add subregions for map: turret plates,{role}_t1_turret, {role}_t2_turret, {role}_inhib_turret, nexus_turret
        self.subregion_cfg = {
            "scoreboard": {
                "KDA": {
                    "thresh_cfg": 110,
                    "ocr_cfg": "--psm 7 -c tessedit_char_whitelist=0123456789/",
                    "thresh_type": "cv.THRESH_BINARY_INV",
                    "mask_coords": {
                        "y1": 0,
                        "y2": 60,
                        "x1": 400,
                        "x2": 600,
                    },
                },
                "Time": {
                    "thresh_cfg": 110,
                    "ocr_cfg": "--psm 7 -c tessedit_char_whitelist=0123456789:",
                    "thresh_type": "cv.THRESH_BINARY_INV",
                    "mask_coords": {
                        "y1": 0,
                        "y2": 60,
                        "x1": 850,
                        "x2": 1000,
                    },
                },
                "CS": {
                    "thresh_cfg": 110,
                    "ocr_cfg": "--psm 7 -c tessedit_char_whitelist=0123456789",
                    "thresh_type": "cv.THRESH_BINARY_INV",
                    "mask_coords": {
                        "y1": 0,
                        "y2": 60,
                        "x1": 670,
                        "x2": 750,
                    },
                },
            },
        }
        # previous read for first reads are always this
        self.prev_ocr = {
            "CS": "0",
            "Kills": 0,
            "Deaths": 0,
            "Assists": 0,
            "Time": "00:00",
        }
        self.VOD_Props = {
            "WIDTH": "",
            "HEIGHT": "",
            "FPS": "",
            "FRAME_COUNT": "",
            "DURATION": "",
            "DES_TIME_INTERVAL": "",
            "FRAME_INDEX": "",
            "RELATIVE_SCALE": "",
        }

        # we didn't keep input paths yet, just in case they're bad
        self.__parse_lol_cfg(lol_cfg_path)
        self.__det_parse_type(realm_path)

        print(f"Parse type determined: {self.parse_type}")

    def __parse_lol_cfg(self, lol_cfg_path):
        """Parses player LoL config file.

        Args:
            lol_cfg_path (str): Path to player LoL cfg file. Can cause errors if from another player.
        """

        config = configparser.ConfigParser()
        if _path_exists(lol_cfg_path):
            # TODO make a for loop of try so that failing one fetch doesn't fail all of them
            try:
                config.read(lol_cfg_path)
                self.lol_cfg["height"] = int(config["General"]["Height"])
                self.lol_cfg["width"] = int(config["General"]["Width"])
                self.lol_cfg["mapscale"] = float(config["HUD"]["MinimapScale"])
                self.lol_cfg["globscale"] = float(config["HUD"]["GlobalScale"])
                self.lol_cfg["mapflipped"] = bool(config["HUD"]["FlipMiniMap"])
                self.lol_cfg["relativeteamcolor"] = str(
                    config["General"]["RelativeTeamColors"]
                )
                # ["HUD"]["ShowSummonerNames"]
                # ["HUD"]["ShowSummonerNamesInScoreboard"]
                # ["HUD"]["NumericCooldownFormat"]

                # if it made it through those steps w/ no error
                self.lol_cfg_path = lol_cfg_path

            except:
                # We've checked that the file exists, so error is likely that a field was missing
                ImportError(
                    "Your congfig file is likely missing some field needed to parse game."
                )

    def __det_parse_type(self, realm_path):
        """Determine appropriate parse type for the realm_path.

        Args:
            realm_path (str): Path to file to be OCR'ed
        """
        # type can only be img, video, or path
        have_reader = cv.haveImageReader(realm_path)

        # check if is img
        if have_reader:
            # file is img
            self.parse_type = np.ndarray
            self.current_img = cv.imread(realm_path)
            self.realm_path = realm_path

        # if not img, give me a video
        elif _path_exists(realm_path):
            self.cap = cv.VideoCapture(realm_path)
            # being redundant here because of a bug w/ opening video in opencv
            self.cap.open(realm_path)

            if self.cap.isOpened():
                # video exists and is openable
                self.parse_type = cv.VideoCapture
                self.realm_path = realm_path

        else:
            return TypeError("File at path -{realm_path}- exists, but won't load")


class VodParse(RealmParser):
    def __init__(
        self,
        realm_path: str,
        input_region: str,
        lol_cfg_path: str,
        role: str,
        des_time_interval=10,
    ):
        """Parse the specified region of a VOD.

        Args:
            realm_path (str): Path to the video or img to be OCR'ed
            input_region (str): Region to parse. Accepted Args: "scoreboard",
            lol_cfg_path (str): Path to player's config file.
            role (str): Any descriptive string. Design intent is to allow unique id of RealmParse objs
            des_time_interval (int, optional): Desired time interval (in seconds) to attempt reads. Defaults to 10.
        """
        # TODO Add capability of parsing specified frame(s)
        self.des_time_interval = des_time_interval

        RealmParser.__init__(
            self,
            realm_path,
            input_region,
            lol_cfg_path,
            role,
        )
        self.parse_vod()

    def __progress_update(self):
        """Prints progress of OCR."""
        print(f"Frame Processed: {self.ocr_out}")

    def __get_VOD_props(self):
        """Assigns properties to the VOD_Props dict if parsing a VOD"""

        self.VOD_Props["WIDTH"] = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.VOD_Props["HEIGHT"] = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        # where possible, stick to integers
        self.VOD_Props["FPS"] = int(self.cap.get(cv.CAP_PROP_FPS))
        self.VOD_Props["FRAME_COUNT"] = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.VOD_Props["DURATION"] = (
            self.VOD_Props["FRAME_COUNT"] / self.VOD_Props["FPS"]
        ) / 60
        self.VOD_Props["DESIRED_TIME_INTERVAL"] = (
            self.des_time_interval * self.VOD_Props["FPS"]
        )
        self.VOD_Props["FRAME_INDEX"] = range(
            0, self.VOD_Props["FRAME_COUNT"], self.VOD_Props["DESIRED_TIME_INTERVAL"]
        )
        self.VOD_Props["RELATIVE_SCALE"] = round(
            (self.VOD_Props["HEIGHT"] * 100 / 2140), 0
        )

    def __get_img_list(self):
        # TODO: find out whether you can get a list of images from a video instead of parsing them one at a time
        pass

    def resize_by_pxpercent(
        self, input_img, desired_px=None, scale_wrt_w_or_h="h", scale_percent=None
    ):
        """Resize an image by pixels or percent, w.r.t. width or height.

        Args:
            input_img (cv.img): cv2 loaded img to resize
            desired_px (int, optional): Pixel size to resize to, can only use if not using percent. Defaults to None.
            scale_wrt_w_or_h (str, optional): Resize with respect to "width" or "height". Defaults to None.
            scale_percent (int, optional): Percent as int to scale to, can only use if not using pixels. Defaults to None.

        Returns:
            cv.img: returns resized cv img
        """
        # TODO rewrite this function such that the required inputs are more clear (they're binary) before implimenting map
        # Note for Reference: shape[0]=height; shape[1]=width
        if scale_wrt_w_or_h == "w":
            scale_percent = desired_px / input_img.shape[1] * 100
        elif scale_wrt_w_or_h == "h":
            scale_percent = desired_px / input_img.shape[0] * 100

        width = int(input_img.shape[1] * scale_percent / 100)
        height = int(input_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv.resize(input_img, dim, interpolation=cv.INTER_AREA)

        return resized

    def get_mask(self, frame, height=None, width=None, subregion=None):
        """Get mask from region_cfg or subregion_cfg.

        Args:
            frame (np.ndarray): Cv img to get mask of.
            subregion (str, optional): Subregion to get mask for. Defaults to None.

        Returns:
            np.ndarray: Returns cropped cv.img
        """

        if not subregion:
            height = frame.shape[0]
            width = frame.shape[1]
            crop_y1 = eval(self.region_cfg[self.input_region]["mask_coords"]["y1"])
            crop_y2 = eval(self.region_cfg[self.input_region]["mask_coords"]["y2"])
            crop_x1 = eval(self.region_cfg[self.input_region]["mask_coords"]["x1"])
            crop_x2 = eval(self.region_cfg[self.input_region]["mask_coords"]["x2"])

        else:
            crop_y1 = self.subregion_cfg[self.input_region][subregion]["mask_coords"][
                "y1"
            ]
            crop_y2 = self.subregion_cfg[self.input_region][subregion]["mask_coords"][
                "y2"
            ]
            crop_x1 = self.subregion_cfg[self.input_region][subregion]["mask_coords"][
                "x1"
            ]
            crop_x2 = self.subregion_cfg[self.input_region][subregion]["mask_coords"][
                "x2"
            ]

        crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]
        return crop

    def __cache_img(self, cvimg: np.ndarray, file_name: str):
        """Store image in local cache.

        Args:
            cvimg (np.ndarray): Image to be cached.
            file_name (str): Name to save file as.
        """
        cv.imwrite(f"realmparser/cache/{file_name}.png", cvimg)
        cached_img = cv.imread(f"realmparser/cache/{file_name}.png")

    def _prep_vod_img(self):
        """Use a series of image transforms to prep img for OCR."""
        # defi7ne image colorspace
        frame_rgb = cv.cvtColor(self.current_frame, cv.COLOR_BGR2RGB)
        frame_grey = cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)

        # mask to field
        region_mask = self.get_mask(frame_grey)

        # pre process field
        resized_region_img = self.resize_by_pxpercent(region_mask, 1000, "w")
        resized_blurred_region_img = cv.GaussianBlur(resized_region_img, (5, 5), 0)
        self.__cache_img(resized_blurred_region_img, self.input_region)

        # If no input subregion, use all of that region's subregions
        subregions = self.subregion_cfg[self.input_region].keys()

        # iterate through subregions
        for subregion in subregions:

            # crop to subregion & cache crop... lol "cache" crop
            # TODO optimize this by returning a list of cropped imgs instead of iteratively cropping
            subregion_mask = self.get_mask(
                resized_blurred_region_img, subregion=subregion
            )
            self.__cache_img(subregion_mask, subregion)
            cached_subregion_mask = cv.imread(f"realmparser/cache/{subregion}.png")

            thresh_cfg = int(
                self.subregion_cfg[self.input_region][subregion]["thresh_cfg"]
            )
            thresh_type = eval(
                self.subregion_cfg[self.input_region][subregion]["thresh_type"]
            )
            # open cache crop (because weird bug w/ cv2)
            ret, threshed_subregion_mask = cv.threshold(
                cached_subregion_mask,
                thresh_cfg,
                255,
                thresh_type,
            )
            subcache_str = f"{subregion}_thresh"
            self.__cache_img(threshed_subregion_mask, subcache_str)

    def parse_vod(self):
        """Parse a video."""

        # TODO once it can parse a whole VOD, add specific frame capabilities
        while self.cap.isOpened():
            # determine properties of VOD
            self.__get_VOD_props()
            self.reader = easyocr.Reader(["en"])
            print("Successfully loaded OCR model")
            print(f"Beginning to parse imgs from {self.realm_path}")

            for self.current_frame_number in self.VOD_Props["FRAME_INDEX"]:
                self.cap.set(1, self.current_frame_number)
                success, self.current_frame = self.cap.read()
                # prep ocr reader

                if success:
                    self._prep_vod_img()
                    self._run_ocr()
                    self.__progress_update()

            self.cap.release()
            total_time = round((time() - START_TIME) / 60, 2)
            print(
                round(self.VOD_Props["DURATION"], 2),
                f"minute VOD parsed in {total_time} minutes. Data collected every {self.des_time_interval} seconds, unless misread.",
            )

    def _run_ocr(self):
        """Run OCR operations."""
        # split cache string for easy parsing later
        self.cache_prefix = "realmparser\\cache\\"
        self.thresh_suffix = "_thresh.png"
        threshed_imgs = glob(f"{self.cache_prefix}*{self.thresh_suffix}")

        # prep read list for input into dict
        self.__ocr_read_list(threshed_imgs)

    def __ocr_read_list(self, ocr_img_paths: list):
        """Perform several list comprehensions & prep ocr read.

        Args:
            ocr_img_paths (list): List of paths to images to be parsed.
        """

        # create list of keys
        ocr_key_list = [
            imgpath.replace(self.cache_prefix, "").replace(self.thresh_suffix, "")
            for imgpath in ocr_img_paths
        ]
        # ocr read imgs
        ocr_read_list = [
            self.reader.readtext(
                imgpath, allowlist=self.region_cfg[self.input_region]["ocr_whitelist"]
            )
            for imgpath in ocr_img_paths
        ]
        # keep only read value
        ocr_read_list = [
            ocr_read[0][-2] if ocr_read != [] else "" for ocr_read in ocr_read_list
        ]

        ocr_read_list.insert(0, self.current_frame_number)
        ocr_key_list.insert(0, "FrameNumber")

        ocr_out = dict(zip(ocr_key_list, ocr_read_list))
        # before doing anything w/ sanity check REMEMBER:
        # OCR output default type is str
        ocr_out = self.__ocr_sanity_check(ocr_out)
        self.ocr_out = ocr_out

        if ocr_out != {}:
            self.output.append(ocr_out)

    def __ocr_sanity_check(self, ocr_reads: dict):
        """Clean output of OCR & only keep good data.

        Args:
            ocr_reads (dict): Dictionary of ocr outputs

        Returns:
            ocr_read: Returns cleaned dictionary. If something misread, returns {}
        """
        # TODO: Rewrite these to pytest format at some point
        # CS TESTS:
        # is cs blank
        if ocr_reads["CS"] == "":
            # if CS is blank, was prev CS 0?
            if self.prev_ocr["CS"] == "0":
                # if prev CS was 0, read is 0
                ocr_reads["CS"] = "0"
            else:
                return {}
        elif type(eval(ocr_reads["CS"])) is int:
            if eval(ocr_reads["CS"]) >= eval(self.prev_ocr["CS"]):
                pass
            else:
                return {}
        else:
            return {}

        # KDA TESTS:
        # has two "/";
        if ocr_reads["KDA"].count("/") == 2:
            # check if it can split w/o making error
            try:
                kda_split = ocr_reads["KDA"].split("/")
                kda_split = [int(val_str) for val_str in kda_split]
            except:
                return {}
            # has 3 vals
            if len(kda_split) == 3:
                # all vals are ints;
                if all([True if type(val) is int else False for val in kda_split]):
                    # prev >= current
                    if (kills := kda_split[0]) >= self.prev_ocr["Kills"]:
                        ocr_reads.pop("KDA")
                    else:
                        return {}

                    if (deaths := kda_split[1]) >= self.prev_ocr["Deaths"]:
                        pass
                    else:
                        return {}

                    if (assists := kda_split[2]) >= self.prev_ocr["Assists"]:
                        pass
                    else:
                        return {}
                    ocr_reads["Kills"] = kills
                    ocr_reads["Deaths"] = deaths
                    ocr_reads["Assists"] = assists
                else:
                    return {}
            else:
                return {}
        else:
            return {}

        # Time TESTS:
        # has one ":" and can be split by it
        if split_time := ocr_reads["Time"].split(":"):
            #  int of len 2 on both sides
            if len(split_time[0]) == 2 and len(split_time[1]) == 2:
                # if current >= prev time

                time_check_passed = datetime.strptime(
                    ocr_reads["Time"], "%M:%S"
                ) >= datetime.strptime(self.prev_ocr["Time"], "%M:%S")

                if time_check_passed:
                    pass
                else:
                    return {}
            else:
                return {}
        else:
            return {}

        self.prev_ocr = ocr_reads

        return ocr_reads
