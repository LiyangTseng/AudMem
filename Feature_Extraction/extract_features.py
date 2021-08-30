import os
import numpy as np
import librosa
from tqdm import tqdm
from PMEmo.features import extract_all_wav_feature, extract_frame_feature, process_dynamic_feature
import arff
import pickle
import sklearn

AUDIO_DIR = '../Experiment_Website/clips'


def extract_chord_features():

    ''' extract choragram and tonnetz from LibROSA'''

    chorma_dir = 'features/chords/chorma'
    if not os.path.exists(chorma_dir):
        os.makedirs(chorma_dir)
    tonnetz_dir = 'features/chords/tonnetz'
    if not os.path.exists(tonnetz_dir):
        os.makedirs(tonnetz_dir)

    for audio_file in tqdm(os.listdir(AUDIO_DIR), desc='Chord Features', leave=True):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        chorma_file = 'chroma_'+ audio_file[:16]
        tonnetz_file = 'tonnetz_'+ audio_file[:16]
        y, sr = librosa.load(audio_path)
        chroma_cqt = librosa.feature.chroma_cens(y=y, sr=sr)
        np.save(os.path.join(chorma_dir, chorma_file), chroma_cqt)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        np.save(os.path.join(tonnetz_dir, tonnetz_file), tonnetz)

    print('chroma extracted from {} and stored at {}'.format(AUDIO_DIR, chorma_dir))
    print('tonnetz extracted from {} and stored at {}'.format(AUDIO_DIR, tonnetz_dir))

def extract_rhythm_features():
    ''' extract rhythm pattern (FA) from MA Toolbox '''
    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # tf = eng.isprime(37)
    # print(tf)

def extract_timbre_features():

    ''' extract MFCC and features related to spectrum shape from LibROSA'''
    
    mfcc_dir = 'features/timbre/mfcc'
    if not os.path.exists(mfcc_dir):
        os.makedirs(mfcc_dir)
    mfcc_delta_dir = 'features/timbre/mfcc_delta'
    if not os.path.exists(mfcc_delta_dir):
        os.makedirs(mfcc_delta_dir)
    mfcc_delta2_dir = 'features/timbre/mfcc_delta2'
    if not os.path.exists(mfcc_delta2_dir):
        os.makedirs(mfcc_delta2_dir)
    spectral_centroid_dir = 'features/timbre/spectral_centroid'
    if not os.path.exists(spectral_centroid_dir):
        os.makedirs(spectral_centroid_dir)
    spectral_bandwidth_dir = 'features/timbre/spectral_bandwidth'
    if not os.path.exists(spectral_bandwidth_dir):
        os.makedirs(spectral_bandwidth_dir)
    spectral_contrast_dir = 'features/timbre/spectral_contrast'
    if not os.path.exists(spectral_contrast_dir):
        os.makedirs(spectral_contrast_dir)
    spectral_flatness_dir = 'features/timbre/spectral_flatness'
    if not os.path.exists(spectral_flatness_dir):
        os.makedirs(spectral_flatness_dir)
    spectral_rolloff_dir = 'features/timbre/spectral_rolloff'
    if not os.path.exists(spectral_rolloff_dir):
        os.makedirs(spectral_rolloff_dir)
     
    for audio_file in tqdm(os.listdir(AUDIO_DIR), desc='Timbre Features', leave=True):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        mfcc_file = 'mfcc_' + audio_file[:16]
        mfcc_delta_file = 'mfccDelta_' + audio_file[:16]
        mfcc_delta2_file = 'mfccDelta2_' + audio_file[:16]
        spectral_centroid_file = 'cent_' + audio_file[:16]
        spectral_bandwidth_file = 'spec_bw_' + audio_file[:16]
        spectral_contrast_file = 'contrast_' + audio_file[:16]
        spectral_flatness_file = 'flatness_' + audio_file[:16]
        spectral_rolloff_file = 'rolloff_' + audio_file[:16]
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.chroma_cens(y=y, sr=sr)
        np.save(os.path.join(mfcc_dir, mfcc_file), mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        np.save(os.path.join(mfcc_delta_dir, mfcc_delta_file), mfcc_delta)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        np.save(os.path.join(mfcc_delta2_dir, mfcc_delta2_file), mfcc_delta2)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        np.save(os.path.join(spectral_centroid_dir, spectral_centroid_file), cent)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        np.save(os.path.join(spectral_bandwidth_dir, spectral_bandwidth_file), spec_bw)
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        np.save(os.path.join(spectral_centroid_dir, spectral_contrast_file), contrast)
        flatness = librosa.feature.spectral_flatness(y=y)
        np.save(os.path.join(spectral_flatness_dir, spectral_flatness_file), flatness)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        np.save(os.path.join(spectral_rolloff_dir, spectral_rolloff_file), rolloff)
        

    print('mfcc extracted from {} and stored at {}'.format(AUDIO_DIR, mfcc_dir))
    print('mfcc_delta extracted from {} and stored at {}'.format(AUDIO_DIR, mfcc_delta_dir))
    print('mfcc_delta2 extracted from {} and stored at {}'.format(AUDIO_DIR, mfcc_delta2_dir))
    print('spectral_centroid extracted from {} and stored at {}'.format(AUDIO_DIR, spectral_centroid_dir))
    print('spectral_bandwidth extracted from {} and stored at {}'.format(AUDIO_DIR, spectral_bandwidth_dir))
    print('spectral_contrast extracted from {} and stored at {}'.format(AUDIO_DIR, spectral_contrast_dir))
    print('spectral_flatness extracted from {} and stored at {}'.format(AUDIO_DIR, spectral_flatness_dir))
    print('spectral_rollof extracted from {} and stored at {}'.format(AUDIO_DIR, spectral_rolloff_dir))

def extract_emotion_features():

    ''' extract predicted valence, arousal from PMEmo '''
    
    def extract_opensmile_features():
        # extract features for PMEmo 
        wavdir = AUDIO_DIR
        opensmiledir = 'opensmile'
        tmp_feature_folder = 'features/emotions/tmp'
        if not os.path.exists(tmp_feature_folder):
            os.makedirs(tmp_feature_folder)
        static_distfile = os.path.join(tmp_feature_folder, "static_features.arff") 
        lld_distdir = os.path.join(tmp_feature_folder, "IS13features_lld")
        dynamic_distdir = os.path.join(tmp_feature_folder, "dynamic_features")
        all_dynamic_distfile = os.path.join(tmp_feature_folder, "dynamic_features.csv")
        delimiter = ";"

        extract_all_wav_feature(wavdir,static_distfile,opensmiledir)
        extract_frame_feature(wavdir,lld_distdir,opensmiledir)
        process_dynamic_feature(lld_distdir,dynamic_distdir,all_dynamic_distfile,delimiter)

    # extract_opensmile_features()


    # predict valence and arousal using PMEmo
    with open('features/emotions/tmp/static_features.arff') as f:
        static_features = arff.load(f)
    static_features = np.array(static_features['data'])[:, 1:-1]
    
    # ref: https://stackoverflow.com/questions/32700797/saving-a-cross-validation-trained-model-in-scikit
    with open(os.path.join('features/emotions/tmp', 'svr_linear_arousal.pkl') , 'rb') as fid:
        arousal_model = pickle.load(fid)
    
    with open(os.path.join('features/emotions/tmp', 'svr_linear_valence.pkl') , 'rb') as fid:
        valence_model = pickle.load(fid)

    predicted_static_arousal = arousal_model.predict(static_features)
    predicted_static_valence = valence_model.predict(static_features)
    print(predicted_static_arousal.shape, predicted_static_valence.shape)
    # TODO: save static arousal & valence, then extract dynamic labels


if __name__ == '__main__':
    # extract_chord_features()
    # extract_timbre_features()
    extract_emotion_features()

    