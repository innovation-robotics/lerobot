from huggingface_hub import HfApi
api = HfApi()

repo_id = "Sokrat2000/multiple_places"

try:
    # Create the v3.0 tag to match your info.json
    api.create_tag(repo_id=repo_id, tag="v3.0", repo_type="dataset")
    print("Tag v3.0 created successfully!")
except Exception as e:
    print(f"Note: {e}")