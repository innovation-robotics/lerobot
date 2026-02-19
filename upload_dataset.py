# import sys
# import os

# # Add the 'src' directory to the path so Python can find 'lerobot'
# sys.path.append(os.path.abspath("src"))

# # In the latest versions, 'common' is often removed from the path
# try:
#     from lerobot.datasets.lerobot_dataset import LeRobotDataset
# except ImportError:
#     from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# # Initialize and push
# dataset = LeRobotDataset(
#     repo_id="waly/new_home_adaptation", 
#     root="/home/ahmed-waly/.cache/huggingface/lerobot"
# )

# dataset.push_to_hub("sokrat2000/new_home_adaptation")

import sys
import os
from pathlib import Path

# Set up paths
repo_root = "/home/ahmed-waly/Repos/lerobot"
sys.path.append(os.path.join(repo_root, "src"))

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Point to your local folder
# NOTE: repo_id here must match the subfolder name inside 'root'
dataset = LeRobotDataset(
    repo_id="waly/new_home_adaptation", 
    root="/home/ahmed-waly/.cache/huggingface/lerobot"
)

# If the __init__ above still tries to connect to the Hub, 
# it's because it's not finding the local files it expects.
# Ensure 'info.json' exists at: 
# /home/ahmed-waly/.cache/huggingface/lerobot/waly/new_home_adaptation/info.json

dataset.push_to_hub("sokrat2000/new_home_adaptation")
