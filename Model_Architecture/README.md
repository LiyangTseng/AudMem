# Model
## Dependency
```

```
## File Usage
- Data augmentation by shifting [original audios](data/raw_audios/original) semitones using [augment_data.py](utils/augment_data.py). The augmentated audios will also be stored at the folder [data/raw_audios](data/raw_audios)
- Extract features from raw audios using [extract_features.py](utils/extract_features.py)
- Hyperparameter configurations saved at [config](config)
- All data (features, labels, raw_audios) saved at [data](data)
- Model weights saved at [weights](weights)
