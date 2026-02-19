# # # from lerobot.datasets.lerobot_dataset import LeRobotDataset
# # # from pathlib import Path
# # # import torch

# # # repo_id = "waly/multiple_places"
# # # local_root = Path("~/.cache/huggingface/lerobot").expanduser()

# # # print(f"--- Integrity Check for: {repo_id} ---")

# # # try:
# # #     ds = LeRobotDataset(repo_id, root=local_root)
    
# # #     # 1. Check Frame Continuity
# # #     # The 'to' of episode N must be exactly one less than the 'from' of episode N+1
# # #     has_gaps = False
# # #     for i in range(len(ds.episode_data_index['from']) - 1):
# # #         end_current = ds.episode_data_index['to'][i]
# # #         start_next = ds.episode_data_index['from'][i+1]
# # #         if start_next != end_current:
# # #             print(f"❌ GAP DETECTED: Episode {i} ends at {end_current}, but Episode {i+1} starts at {start_next}")
# # #             has_gaps = True
    
# # #     if not has_gaps:
# # #         print("✅ Index Continuity: Perfect. No missing frames between episodes.")

# # #     # 2. Check Physical File Bounds
# # #     # Try to load the very first and very last frame to ensure parquet files aren't truncated
# # #     try:
# # #         first_frame = ds[0]
# # #         last_frame = ds[ds.num_frames - 1]
# # #         print(f"✅ Data Access: Successfully read first and last frames (Total: {ds.num_frames}).")
# # #     except IndexError as e:
# # #         print(f"❌ BOUNDS ERROR: Could not access all frames. {e}")

# # #     # 3. Video Backend Check
# # #     # This checks if the video files exist for the first episode
# # #     video_path = Path(ds.root) / "videos"
# # #     if video_path.exists():
# # #         print(f"✅ Video Folder: Found at {video_path}")
# # #     else:
# # #         print("⚠️ Video Folder: Not found (Only okay if you are using internal image storage).")

# # #     print("\n--- Summary ---")
# # #     if not has_gaps and ds.num_frames > 0:
# # #         print("PASSED: This dataset is safe for training.")
# # #     else:
# # #         print("FAILED: Integrity issues found.")

# # # except Exception as e:
# # #     print(f"❌ CRITICAL LOAD ERROR: {e}")

# # from lerobot.datasets.lerobot_dataset import LeRobotDataset
# # from pathlib import Path
# # import torch

# # repo_id = "waly/multiple_places_fix"
# # local_root = Path("~/.cache/huggingface/lerobot").expanduser()

# # print(f"--- Integrity Check (Offline Mode) for: {repo_id} ---")

# # try:
# #     ds = LeRobotDataset(
# #         repo_id, 
# #         root=local_root, 
# #         local_files_only=True
# #     )
    
# #     # 1. Check Frame Continuity
# #     # The 'to' of episode N must be exactly one less than the 'from' of episode N+1
# #     has_gaps = False
# #     for i in range(len(ds.episode_data_index['from']) - 1):
# #         end_current = ds.episode_data_index['to'][i]
# #         start_next = ds.episode_data_index['from'][i+1]
# #         if start_next != end_current:
# #             print(f"❌ GAP DETECTED: Episode {i} ends at {end_current}, but Episode {i+1} starts at {start_next}")
# #             has_gaps = True
    
# #     if not has_gaps:
# #         print("✅ Index Continuity: Perfect. No missing frames between episodes.")

# #     # 2. Check Physical File Bounds
# #     # Try to load the very first and very last frame to ensure parquet files aren't truncated
# #     try:
# #         first_frame = ds[0]
# #         last_frame = ds[ds.num_frames - 1]
# #         print(f"✅ Data Access: Successfully read first and last frames (Total: {ds.num_frames}).")
# #     except IndexError as e:
# #         print(f"❌ BOUNDS ERROR: Could not access all frames. {e}")

# #     # 3. Video Backend Check
# #     # This checks if the video files exist for the first episode
# #     video_path = Path(ds.root) / "videos"
# #     if video_path.exists():
# #         print(f"✅ Video Folder: Found at {video_path}")
# #     else:
# #         print("⚠️ Video Folder: Not found (Only okay if you are using internal image storage).")

# #     print("\n--- Summary ---")
# #     if not has_gaps and ds.num_frames > 0:
# #         print("PASSED: This dataset is safe for training.")
# #     else:
# #         print("FAILED: Integrity issues found.")

# # except Exception as e:
# #     print(f"❌ CRITICAL LOAD ERROR: {e}")

# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from pathlib import Path

# # Use the exact folder you created
# fixed_path = Path("~/.cache/huggingface/lerobot/waly/multiple_places").expanduser()

# print(f"--- Integrity Check for Local Folder ---")
# print(f"Path: {fixed_path}")

# try:
#     # We pass the path as 'root' and set repo_id to None 
#     # to stop it from trying to connect to the internet.
#     ds = LeRobotDataset(repo_id=None, root=fixed_path)
    
#     # 1. Check Frame Continuity
#     has_gaps = False
#     for i in range(len(ds.episode_data_index['from']) - 1):
#         end_current = ds.episode_data_index['to'][i]
#         start_next = ds.episode_data_index['from'][i+1]
#         if start_next != end_current:
#             print(f"❌ GAP DETECTED: Episode {i} ends at {end_current}, but Episode {i+1} starts at {start_next}")
#             has_gaps = True
    
#     if not has_gaps:
#         print("✅ Index Continuity: Perfect. All frames are sequential.")

#     # 2. Check Physical Data Access
#     try:
#         # Check first, middle, and last
#         _ = ds[0]
#         _ = ds[ds.num_frames // 2]
#         _ = ds[ds.num_frames - 1]
#         print(f"✅ Data Access: Successfully read samples across the dataset (Total: {ds.num_frames} frames).")
#     except Exception as e:
#         print(f"❌ DATA ERROR: Could not read frames. {e}")

#     print("\n--- Summary ---")
#     if not has_gaps and ds.num_frames > 0:
#         print("RESULT: PASSED. This folder is a healthy LeRobot dataset.")
#     else:
#         print("RESULT: FAILED.")

# except Exception as e:
#     print(f"❌ FOLDER ERROR: Could not load dataset from path. {e}")

import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configuration
repo_id = "waly/multiple_places"
local_root = Path("~/.cache/huggingface/lerobot").expanduser()
fixed_path = Path("~/.cache/huggingface/lerobot/waly/multiple_places").expanduser()

print(f"--- LeRobot v3.0 Integrity Check ---")
print(f"Target: {local_root / repo_id}")

try:
    # 1. Load the dataset (using root to ensure it finds your local 'fix' folder)
    # We use repo_id=repo_id because v3.0 metadata handles the pathing internally
    ds = LeRobotDataset(repo_id=None, root=fixed_path)
    
    # In v3.0, episode metadata is under ds.meta.episodes
    # It is usually a list of dicts or a pandas DataFrame
    episodes = ds.meta.episodes
    
    print(f"Found {len(episodes)} episodes in metadata.")
    
    # 2. Check Index Continuity
    # We check that episode[N].to == episode[N+1].from
    has_gaps = False
    for i in range(len(episodes) - 1):
        curr_ep = episodes[i]
        next_ep = episodes[i+1]
        
        curr_to = curr_ep["dataset_to_index"]
        next_from = next_ep["dataset_from_index"]
        
        if curr_to != next_from:
            print(f"❌ GAP FOUND between Ep {i} and Ep {i+1}!")
            print(f"   Ep {i} ends at index: {curr_to}")
            print(f"   Ep {i+1} starts at index: {next_from}")
            has_gaps = True

    if not has_gaps:
        print("✅ Index Continuity: Perfect. All episodes connect sequentially.")

    # 3. Check Physical Access (The Parquet Check)
    # We try to grab a frame from the start, middle, and end
    try:
        total_frames = ds.meta.total_frames
        i1 = ds[0]
        i2 = ds[total_frames // 2]
        i3 = ds[total_frames - 1]
        print(f"✅ Data Access: Successfully read samples (Total frames: {total_frames}).")
    except Exception as e:
        print(f"❌ DATA ACCESS ERROR: The files on disk don't match the metadata. Error: {e}")

    # 4. Video Path Check
    if len(ds.meta.video_keys) > 0:
        for v_key in ds.meta.video_keys:
            # Check if the first video file actually exists
            try:
                v_path = ds.meta.get_video_file_path(0, v_key)
                if (ds.root / v_path).exists():
                    print(f"✅ Video Check: '{v_key}' files found.")
                else:
                    print(f"⚠️ Video Warning: Metadata points to {v_path}, but file is missing.")
            except Exception:
                print(f"⚠️ Video Warning: Could not resolve path for {v_key}.")

    print("\n--- Final Result ---")
    if not has_gaps:
        print("PASSED: The dataset index is healthy.")
    else:
        print("FAILED: The dataset is still corrupted.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not initialize dataset. {e}")