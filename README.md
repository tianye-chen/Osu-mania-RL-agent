For CSE573 final project at UB Fall 2024

## YOLOv5n (Object Identification) training setup

- Download osu!lazer from the official site
- Apply the custom skin by running the .osk file
- Set background dim to 100% in settings
- Select an appropriate beatmap preferably a decent distribution of normal and hold notes over multiple beatmaps
- Select a replay of the beatmap and run vision.ipynb, a 60 second recording at 1 FPS will be saved at /video_out/
- Run dataset_process.ipynb to split each recording frame into an image saved at /frames/unlabeled/{idx}
- Upload the /frames/unlabeled/{idx} folder to Roboflow
- Assign labeling task and label each element by "note", "start_hold", "end_hold"
  - "note" and "start_hold" should be at least 30% visible to be considered
  - The tail end of "end_hold" should be entirely visible to be considered

## Customized osu!lazer for score retrieval via sockets
- Clone official osu!lazer client from https://github.com/ppy/osu
- Replace "\osu\osu.Game.Rulesets.Mania\Scoring\ManiaScoreProcessor.cs" with \misc\ManiaScoreProcessor.cs
- Navigate to \osu\ in a terminal and start build using dotnet run --project osu.Desktop
- Use helper.SocketListener class to retrieve note hit judgements
- See socket_test.ipynb for example on SocketListener
