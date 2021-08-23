# Feature Extraction
## File Usage
- [experimentData_beta.csv](experimentData_beta.csv): experiment data exported from MySQL database. The data contains 4 fields as listed below: In particular, "audioOrder" is formatted as a string of 94-number-order separated by comma, whereas "userReponse" is formatted as a string of 162-number-reponse separated by comma.
```
    | updateTime | userEmail | audioOrder | userResponse |
```
- [extract_labels.py](extract_labels.py): python script for extracting labels. The file takes [experimentData_beta.csv](experimentData_beta.csv) as input and will output labels at [labels/track_memorability.csv](labels/track_memorability.csv). Note that the labels are formatted as below:
```
    | tracktrack | repeat_interval:memorized |
```
- [extract_features.py](extract_features.py): extract features from audio files. Features including chomagram(shape: 12, 646), tonnetz(shape: 6, 646), fluctation pattern, mfcc(12, 646), spectrum centroid(1, 646), spectrum bandwidth(1, 646), spectrum contrast(7, 646), spectrum flatness(1, 646), spectrum rolloff(1, 646), emotion.
- [track_index.txt](track_index.txt):  a look-up table that stores the relation between audio filename and index which is used in "AudioOrder" in [experimentData_beta.csv](experimentData_beta.csv).