# RealmParser
A python module using OCR to processing first-person League of Legends videos (ideally for Tournament Realm games)

# Install
Download repository and use run.py

Python Package Dependencies:
 - opencv-python
 - easyocr
 - numpy
 - pprint
 - pandas (not necessary, but *highly* recommended)

# Usage
The most important thing that you'll need are:
- a video to parse
- the game.cfg file of the player whose video is being parsed

```python
import realmparser
import pandas as pd

video_name = "mid"
mid_realm = realmparser.VodParse(
    realm_path=f"{video_name}.mp4",
    input_region="scoreboard",
    lol_cfg_path="game.cfg",
    role="MID",
    des_time_interval=10,

)


df = pd.DataFrame.from_dict(mid_realm.output)
df.to_pickle("midrealm_data.pkl")
df.to_csv("midrealm_data.csv")
```
**Time Required Benchmarks:**
![30sresult](https://i.imgur.com/vSm4aWY.png)
![5sresult](https://i.imgur.com/hCZJvNP.png)

# TODO

**1. New regions and subregions:**

    - Map: TurretPlates, TurretStatus
    - Scoreboard: RedTeamKills, BlueTeamKills
    - ChampHUD: ChampionPlayed, GoldInHand, TotalGold, ItemsBuilt

**2. Restructure output to conform with the [loldto](https://pypi.org/project/lol-dto/):**

    game: dict
    ├── sources: dict
    ├── teams: dict
    |   ├── uniqueIdentifiers: dict
    │   ├── bans: list
    │   ├── monstersKills: list
    │   ├── buildingsKills: list
    │   └── players: list
    │       ├── uniqueIdentifiers: dict
    │       ├── endOfGameStats: dict
    │       │   └── items: list
    │       ├── summonersSpells: list
    │       ├── runes: list
    │       ├── snapshots: list
    │       ├── itemsEvents: list
    │       ├── wardsEvents: list
    │       └── skillsLevelUpEvents: list
    ├── kills: list
    └── picksBans: list

**3. Determine whether list of images can be pulled from a VOD to use list comprehension instead of  a for loop.**