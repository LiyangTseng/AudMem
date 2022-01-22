# Model
## Dependency
To run the codes, the following packages are needed. It is suggested to create a new virtual environment and run <code>pip install -r [requirements.txt](requirements.txt)</code> to install all packages.
```
tqdm==4.28.1
git+https://github.com/santi-pdp/ahoproc_tools.git
numpy==1.17
scipy==1.1.0
librosa==0.6.3
SoundFile==0.10.2
torchaudio==0.4.0
# torchvision==0.5.0
pysptk==0.1.16
matplotlib==3.0.2
python_speech_features==0.6
scikit_learn==0.20.3
tensorboardX==1.6
webrtcvad==2.0.10
cupy-cuda112
pynvrtc==8.0 
git+https://github.com/salesforce/pytorch-qrnn
git+https://github.com/detly/gammatone
git+https://github.com/pswietojanski/kaldi-io-for-python.git
```
Run the following code in [root folder](.) to enable sibling package imports. This is related to [setup.py](setup.py).
```
pip install -e .
```
## File Usage
## General Function
- Hyperparameter configurations saved at [config](config).
- All data (features, labels, raw_audios) saved at [data](data).
- Model weights saved at [weights](weights).
- [train.py](train.py)/[test.py](test.py) for model training/inferencing wrappers.
### Handcrafted Features Model
- Data augmentation by shifting [original audios](data/raw_audios/original) semitones using [augment_data.py](utils/augment_data.py). The augmentated audios will also be stored at the folder [data/raw_audios](data/raw_audios).
- Extract features from raw audios using [extract_features.py](utils/extract_features.py).

### End-to-End Model
#### Pase/Pase+ Encoder
- Set up audioset for pre-training pase using [prepare_audioset.py](utils/prepare_audioset.py).
- Format data configuration to pase-compatible form using [generate_data_cfg.py](utils/generate_data_cfg.py), this is modified from [PASE](https://github.com/santi-pdp/pase/blob/master/unsupervised_data_cfg_librispeech.py).
- Use [make_trainset_statistics.py](utils/make_trainset_statistics.py) to compute data normalization statistics for pretext workers.