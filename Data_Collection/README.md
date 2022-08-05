# Data Collection
## Goal
- To collect appropriate music samples as materials in *[The Music Memory Game](../Experiment_Website/)*
## How it is done
- To avoid bias, Use [Youtube API](https://developers.google.com/youtube/v3/docs/search/list) to collect videos in the category of "music"
- Manuel filter videos that not realy have content of music (e.g., interviews, unboxing musical gadgets)
- Apply pilot study to make sure the collected audios are not that popular (therefore can be used as materials in memory game)
- Apply appropriate truncation and stretching to the audios

## File Usage
> Note that codes in this directory migth be a little bit messy
- [collect_yt_data.py](collect_yt_data.py): get random music-related videos via Youtube API, then store the results to [yt_info.csv](yt_info.csv).
- [brows_all.py](brows_all.py): a program to open all youtube links one by one automatically, and is meant to be used in the process of filtering unwanted music.
- [plot_statistics.py](plot_statistics.py): plot histograms of both viewCounts and valid_viewCounts. 
- [download_audios.py](download_audios.py): first choose evenly distributed data from [yt_info.csv](yt_info.csv) with respect to views and then them as audios.
- [detect_chorus.py](detect_chorus.py): use [pychorus](https://github.com/vivjay30/pychorus) to roughly detect chorus location, the result would be stored at [intro_locations.csv](intro_locations.csv).
- [extract_clips.py](extract_clips.py): extract chorus clips from raw audio using either the chorus locations stored in [intro_locations.csv](intro_locations.csv) or the [audacity](https://www.audacityteam.org/) program to determine the locations on th fly. 
- [normalize_clip.py](normalize_clip.py): used to normalize all volumns and strech the audios to desired length.
