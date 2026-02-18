from pathlib import Path
import json
path = Path("/Users/vyomagarwal/Documents/projects/sound_localization/VGGSound/audioset_data/processed")
c =0 
videos = [v for v in path.iterdir()]
for video in videos:
    video = video / "result.json"
    with open(video, 'r') as f:
        data = json.load(f)
    if data["visual_motion"]["left"] > data["visual_motion"]["right"]:
        motions = data['audio_spatial']["dominant_side"]
        if motions != "left":
            print(video)
            c+=1
    else:
        motions = data['audio_spatial']["dominant_side"]
        if motions != "right":
            print(video)
            c+=1
print(c)
