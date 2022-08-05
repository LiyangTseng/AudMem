# Model Architecture
> This project is written using [this template](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)
## Goal 
Use the labeled [music memorabilty dataset](https://drive.google.com/file/d/16A_3x1FhWq76HpW2Sq5t7-BQzLdnkfAI/view?usp=sharing) from the interactive [Music Memory Game](../Experiment_Website/) to train music memorability prediction
## Implementation (Proposed Models)
1. Explainable handcraft features + SVR
2. Explainable handcraft features + MLP
3. Mel-spectrogram + [Self-Supervised Audio Spectrogram Transformer (SSAST)](https://drive.google.com/file/d/16A_3x1FhWq76HpW2Sq5t7-BQzLdnkfAI/view?usp=sharing)

## File Usage
### General Function
- Can reference [original template](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch) for overview
- Hyperparameter configurations saved at [config](config).
- All data (features, labels, raw_audios) saved at [data](data).
- Model weights saved at [weights](weights).
- [train.py](train.py)/[test.py](test.py) for model training/inferencing wrappers.
- Baseline representations are stored in [baseline_representation.zip](/baseline_representation.zip)
### Feature Engineering Model
- Data augmentation by shifting [original audios](data/raw_audios/original) semitones using [augment_data.py](utils/augment_data.py). The augmentated audios will also be stored at the folder [data/raw_audios](data/raw_audios).
- Note that the *explainable handcraft features (EHC features)* are extracted in [extract_segments_features](extract_segments_features) (after split five-second clips to 9 one-second clips), and the features are stored in [data.csv](data/data.csv).
- Train models using [run_mlp_k_fold.sh](/run_mlp_k_fold.sh) and [run_svr_k_fold.sh](/run_svr_k_fold.sh)
### End-to-End Feature Learning Model
#### SSAST
- Fine-tuned SSAST on our music memorability prediction task
- Train model using [run_ssast_k_fold.sh](/run_ssast_k_fold.sh)
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
