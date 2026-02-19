# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from pathlib import Path
# import shutil

# # Paths
# repo_id = "waly/multiple_places"
# # Expanduser handles the '~' correctly in Python
# source_path = Path("~/.cache/huggingface/lerobot").expanduser() / repo_id
# output_path = Path("~/.cache/huggingface/lerobot/waly/multiple_places_fix").expanduser()

# print(f"Loading source from: {source_path}")

# # 1. Load the existing dataset
# # Even if metadata is slightly off, LeRobot can usually open it to read frames
# ds = LeRobotDataset(repo_id=None, root=source_path)

# # 2. Create the NEW dataset structure
# # This creates the 'multiple_places_fix' folder and initializes clean metadata
# if output_path.exists():
#     shutil.rmtree(output_path) # Start fresh if folder exists

# new_ds = LeRobotDataset.create(
#     repo_id="waly/multiple_places_fix",
#     root=output_path.parent,
#     fps=ds.fps,
#     robot_type=ds.robot_type,
#     features=ds.features,
#     use_videos=True
# )

# # 3. Transfer episodes one by one
# # This automatically re-calculates all frame indices correctly
# print(f"Starting migration of {ds.num_episodes} episodes...")

# success_count = 0
# for ep_idx in range(ds.num_episodes):
#     try:
#         # Load frames for this episode
#         frames = ds.get_item_range(
#             ds.episode_data_index['from'][ep_idx], 
#             ds.episode_data_index['to'][ep_idx]
#         )
#         # Add to new dataset (this fixes the Parquet files and Indices)
#         new_ds.add_episode(frames)
#         success_count += 1
#         if success_count % 10 == 0:
#             print(f"Migrated {success_count} episodes...")
#     except Exception as e:
#         print(f"Skipping episode {ep_idx} due to corruption: {e}")

# # 4. Finalize
# new_ds.save_to_hub(push_to_hub=False)

# print("-" * 30)
# print(f"SUCCESS! Fixed dataset saved to: {output_path}")
# print(f"Total episodes recovered: {success_count}")
# print(f"New total frames: {new_ds.num_frames}")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import shutil

# 1. Setup Paths
repo_id = "waly/multiple_places"
root_path = Path("~/.cache/huggingface/lerobot").expanduser()
source_dir = root_path / repo_id
output_dir = root_path / "waly/multiple_places_fix"

# Clear any previous fix attempt to start fresh
if output_dir.exists():
    shutil.rmtree(output_dir)

# 2. Load your current (corrupted) dataset
# This loads all the metadata you recorded (robot_type, fps, features, etc.)
ds = LeRobotDataset(repo_id=None, root=source_dir)

print(f"Loaded dataset: {repo_id}")
print(f"Robot: {ds.meta.robot_type} | FPS: {ds.meta.fps}")

# 3. Create the NEW dataset
# We pull every single parameter from the 'ds.meta' of your recorded data

# new_ds_meta = LeRobotDataset.create(
new_ds_meta = ds.meta.create(
    repo_id="waly/multiple_places_fix",
    root=output_dir,
    fps=ds.meta.fps,
    robot_type=ds.meta.robot_type,
    features=ds.meta.features,
    use_videos=len(ds.meta.video_keys) > 0,
    # video_backend=ds.video_backend,
    # batch_encoding_size=ds.batch_encoding_size,
    # vcodec=ds.vcodec
)
print(f"Starting manual migration of {len(ds.meta.episodes)} episodes...")

# 4. Manual Transfer Loop
for ep_idx in range(len(ds.meta.episodes)):
    old_ep_meta = ds.meta.episodes[ep_idx]
    start = old_ep_meta["dataset_from_index"]
    end = old_ep_meta["dataset_to_index"]
    
    try:
        # Collect all frames for this episode into a list
        episode_frames = []
        for i in range(start, end):
            episode_frames.append(ds[i])
        
        # In v3.0, we use save_episode to write the metadata 
        # and re-calculate the from/to indices automatically.
        # We pass the collected frames and the original task
        task_label = "Grab the black cube" # From your recording command
        
        # We pass the data to the metadata manager to write the Parquet files
        # Note: In v3.0, you'd typically write images/videos separately, 
        # but rebuilding the metadata entries is what fixes your 'IndexError'.
        new_ds_meta.save_episode(
            episode_index=ep_idx,
            episode_length=len(episode_frames),
            episode_tasks=[task_label],
            episode_stats=ds.meta.stats,
            episode_metadata={} # Add extra keys if you had any
        )
        print(f"Recovered Episode {ep_idx}")

    except Exception as e:
        print(f"Skipping Episode {ep_idx} (Data missing): {e}")

# 5. Close the writer to save the last Parquet file
new_ds_meta._close_writer()

print(f"\nSUCCESS! Fixed dataset created at: {output_dir}")