# Data Collection
## File Usage
- *collect_yt_data.py*: get random music-related videos via Youtube API, then store the results to *yt_info.csv*.
- *brows_all.py*: a program to open all youtube links one by one automatically, and is meant to be used in the process of filtering unwanted music.
- *plot_statistics.py*: plot histograms of both viewCounts and valid_viewCounts. 
- *download_audios.py*: first choose evenly distributed data from *yt_info.csv* with respect to views and then them as audios.
- *detect_chorus.py*: use [pychorus](https://github.com/vivjay30/pychorus) to roughly detect chorus location, the result would be stored at *chorus_location.csv*.
- *extract_chorus.py*: extract chorus clips from raw audio using either the chorus locations stored in *chorus_location.csv* or the [audacity](https://www.audacityteam.org/) program to determine the locations on th fly. 
