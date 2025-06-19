## Using the vision_auto.py script
- Clone official osu!lazer repository from https://github.com/ppy/osu
- Install .NET SDK from https://dotnet.microsoft.com/en-us/download
- Replace "\osu\osu.Game.Rulesets.Mania\Scoring\ManiaScoreProcessor.cs" with "\misc\ManiaScoreProcessor.cs"
- Navigate to \osu\ in a terminal and start local development build using dotnet run --project osu.Desktop
- Apply the provided custom game textures by running the ./misc/osu_skin.osk file
- Run the vision_auto.py file, optional -verbose flag if you wish to see keypresses
- Navigate to the mania game mode and play a song
