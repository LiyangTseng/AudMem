## Add Zero-crossings & genre tagging features to data.csv
 data.csv from [data.csv](https://github.com/LiyangTseng/MusicMem/blob/master/Model_Architecture/data/data.csv), including the initial features we come up with for predicting music memorability. 
> features including Harmony(chords), Rhythm (bpm), Timbre, Mood (emotions)

<br><br>

After several surveys, we decided to extend 2 features, Zero Crossing and Genre taggings.
* step 1: [truncation.py](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/truncation.py)
> truncate 5-second raw audios to 1-second clips if not preprocessed before.
* step 2: [pitch_shift.py](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/pitch_shift.py)
> shift pitch for each 1-second clip if not preprocessed before.
* step 3: [zero_crossing_rate.ipynb](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/zcr/zero%20crossing%20rate.ipynb)
> find zero crossings features and save them into [zcr_df](https://github.com/LiyangTseng/MusicMem/tree/master/Feature_Extraction/new_feature/zcr/zcr_df), there are 3 versions for calculating zcr features for (1) raw audios, (2) truncated raw audios with pitch shifting, (3) truncated raw audios without pitch shifting.
* step 4: [pann_tagging](https://github.com/LiyangTseng/MusicMem/tree/master/Feature_Extraction/new_feature/pann_tagging)
> find genre tagging through the code provided by [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), follow the instruction shows in ["Audio tagging using pretrained models"](https://github.com/qiuqiangkong/audioset_tagging_cnn?tab=readme-ov-file#audio-tagging-using-pretrained-models) to create [original_tag.csv](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/pann_tagging/original_tag.csv) for raw audios (5-second ones, since 1-second clips are too short for genre distinguishing)
> 
> then merged genre taggings for raw audios and their truncated/truncated+pitch shifted transforms together by [tagging.ipynb](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/pann_tagging/tagging.ipynb)
* step 5: [feature_append.ipynb](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/feature%20append.ipynb)
> merge zero crossings and genre taggings (2 featues) and the initial features, get [data_newFeatures.csv](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/feature%20selection/data_newFeature.csv)
* step 6: [feature_selection.py](https://github.com/LiyangTseng/MusicMem/blob/master/Feature_Extraction/new_feature/feature%20selection/feature%20selection.py)
> reducing features to k=25 items.