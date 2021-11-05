import os
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from utils.features import extract_all_wav_feature, extract_frame_feature, process_dynamic_feature
import arff
import pickle
from sklearn.preprocessing import StandardScaler
'''
    this file is the same as the one in Feature_Extraction/extract_features.py
'''

AUDIO_DIR = '../Experiment_Website/clips'


def extract_chord_features():

    ''' extract choragram and tonnetz from LibROSA'''

    chroma_dir = 'features/chords/chroma'
    if not os.path.exists(chroma_dir):
        os.makedirs(chroma_dir)
    tonnetz_dir = 'features/chords/tonnetz'
    if not os.path.exists(tonnetz_dir):
        os.makedirs(tonnetz_dir)

    for audio_file in tqdm(os.listdir(AUDIO_DIR), desc='Chord Features', leave=True):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        chroma_path = os.path.join(chroma_dir, 'chroma_'+ audio_file.replace(".wav", ".npy")) 
        tonnetz_path =  os.path.join(tonnetz_dir, 'tonnetz_'+ audio_file.replace(".wav", ".npy"))
        if os.path.exists(chroma_path) and os.path.exists(tonnetz_path):
            continue
        else:
            y, sr = librosa.load(audio_path)
            if not os.path.exists(chroma_path):
                chroma_cqt = librosa.feature.chroma_cens(y=y, sr=sr)
                np.save(chroma_path, chroma_cqt)
            if not os.path.exists(tonnetz_path):
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                np.save(tonnetz_path, tonnetz)

    print('chroma features of {} saved at {}'.format(AUDIO_DIR, chroma_dir))
    print('tonnetz features of {} saved at {}'.format(AUDIO_DIR, tonnetz_dir))

def extract_rhythm_features():
    
    ''' extract tempogram from libROSA '''
    
    tempogram_dir = 'features/rhythms/tempogram'
    if not os.path.exists(tempogram_dir):
        os.makedirs(tempogram_dir)


    for audio_file in tqdm(os.listdir(AUDIO_DIR), desc='Rhythm Features', leave=True):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        tempogram_path = os.path.join(tempogram_dir, 'tempogram_'+ audio_file.replace(".wav", ".npy"))
        if os.path.exists(tempogram_path):
            continue

        y, sr = librosa.load(audio_path)
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        np.save(tempogram_path, tempogram)

    print('tempogram features of {} saved at {}'.format(AUDIO_DIR, tempogram_dir))
        
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
        mfcc_path = os.path.join(mfcc_dir, 'mfcc_' + audio_file.replace(".wav", ".npy"))
        mfcc_delta_path = os.path.join(mfcc_delta_dir, 'mfcc_delta_' + audio_file.replace(".wav", ".npy"))
        mfcc_delta2_path = os.path.join(mfcc_delta2_dir, 'mfcc_delta2_' + audio_file.replace(".wav", ".npy"))
        spectral_centroid_path = os.path.join(spectral_centroid_dir, 'spectral_centroid_' + audio_file.replace(".wav", ".npy"))
        spectral_bandwidth_path = os.path.join(spectral_bandwidth_dir, 'spectral_bandwidth_' + audio_file.replace(".wav", ".npy"))
        spectral_contrast_path = os.path.join(spectral_contrast_dir, 'spectral_contrast_' + audio_file.replace(".wav", ".npy"))
        spectral_flatness_path = os.path.join(spectral_flatness_dir, 'spectral_flatness_' + audio_file.replace(".wav", ".npy"))
        spectral_rolloff_path = os.path.join(spectral_rolloff_dir, 'spectral_rolloff_' + audio_file.replace(".wav", ".npy"))

        if os.path.exists(mfcc_path) and os.path.exists(mfcc_delta_path) and os.path.exists(mfcc_delta2_path) and \
            os.path.exists(spectral_centroid_path) and os.path.exists(spectral_bandwidth_path) and os.path.exists(spectral_contrast_path) and \
            os.path.exists(spectral_flatness_path) and os.path.exists(spectral_rolloff_path):
            continue
        else:
            y, sr = librosa.load(audio_path)
            if not os.path.exists(mfcc_path):
                mfcc = librosa.feature.chroma_cens(y=y, sr=sr)
                np.save(mfcc_path, mfcc)
            if not os.path.exists(mfcc_delta_path):
                mfcc_delta = librosa.feature.delta(mfcc)
                np.save(mfcc_delta_path, mfcc_delta)
            if not os.path.exists(mfcc_delta2_path):
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                np.save(mfcc_delta2_path, mfcc_delta2)
            if not os.path.exists(spectral_centroid_path):
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                np.save(spectral_centroid_path, cent)
            if not os.path.exists(spectral_bandwidth_path):
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                np.save(spectral_bandwidth_path, spec_bw)
            if not os.path.exists(spectral_contrast_path):
                S = np.abs(librosa.stft(y))
                contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
                np.save(spectral_contrast_path, contrast)
            if not os.path.exists(spectral_flatness_path):
                flatness = librosa.feature.spectral_flatness(y=y)
                np.save(spectral_flatness_path, flatness)
            if not os.path.exists(spectral_rolloff_path):       
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                np.save(spectral_rolloff_path, rolloff)
        

    print('mfcc features of {} saved at {}'.format(AUDIO_DIR, mfcc_dir))
    print('mfcc_delta features of {} saved at {}'.format(AUDIO_DIR, mfcc_delta_dir))
    print('mfcc_delta2 features of {} saved at {}'.format(AUDIO_DIR, mfcc_delta2_dir))
    print('spectral_centroid features of {} saved at {}'.format(AUDIO_DIR, spectral_centroid_dir))
    print('spectral_bandwidth features of {} saved at {}'.format(AUDIO_DIR, spectral_bandwidth_dir))
    print('spectral_contrast features of {} saved at {}'.format(AUDIO_DIR, spectral_contrast_dir))
    print('spectral_flatness features of {} saved at {}'.format(AUDIO_DIR, spectral_flatness_dir))
    print('spectral_rolloff features of {} saved at {}'.format(AUDIO_DIR, spectral_rolloff_dir))

def extract_emotion_features(open_opensmile = False):

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

    if open_opensmile:
        extract_opensmile_features()


    ''' predict static arousal and valence using PMEmo '''
    with open('features/emotions/tmp/static_features.arff') as f:
        arrf_data = arff.load(f)
    static_features = np.array(arrf_data['data'])[:, 1:-1]
    scaler = StandardScaler().fit(static_features)
    scaled_dynamic_features = scaler.transform(static_features)

    audio_filenames = [data[0][:-4] for data in arrf_data['data']]

    # load saved model ref: https://stackoverflow.com/questions/32700797/saving-a-cross-validation-trained-model-in-scikit
    with open('svr_linear_static_arousal.pkl' , 'rb') as fid:
        static_arousal_model = pickle.load(fid)  
    with open('svr_linear_static_valence.pkl' , 'rb') as fid:
        static_valence_model = pickle.load(fid)

    predicted_static_arousal = static_arousal_model.predict(scaled_dynamic_features)
    predicted_static_valence = static_valence_model.predict(scaled_dynamic_features)

    ''' predict dynamic arousal and valence using PMEmo '''
    csv_data = pd.read_csv('features/emotions/tmp/dynamic_features.csv')
    dynamic_features = csv_data[csv_data.columns[2:262]]
    scaler = StandardScaler().fit(dynamic_features)
    scaled_dynamic_features = scaler.transform(dynamic_features)
    
    with open('svr_rbf_dynamic_arousal.pkl' , 'rb') as fid:
        dynamic_arousal_model = pickle.load(fid)
    with open('svr_rbf_dynamic_valence.pkl' , 'rb') as fid:
        dynamic_valence_model = pickle.load(fid)

    predicted_dynamic_arousal = dynamic_arousal_model.predict(scaled_dynamic_features)
    predicted_dynamic_valence = dynamic_valence_model.predict(scaled_dynamic_features)

    ''' write static/dynamic arousal/valence to file '''
    static_arousal_dir = 'features/emotions/static_arousal'
    if not os.path.exists(static_arousal_dir):
        os.makedirs(static_arousal_dir)
    static_valence_dir = 'features/emotions/static_valence'
    if not os.path.exists(static_valence_dir):
        os.makedirs(static_valence_dir)
    dynamic_arousal_dir = 'features/emotions/dynamic_arousal'
    if not os.path.exists(dynamic_arousal_dir):
        os.makedirs(dynamic_arousal_dir)
    dynamic_valence_dir = 'features/emotions/dynamic_valence'
    if not os.path.exists(dynamic_valence_dir):
        os.makedirs(dynamic_valence_dir)

    single_dynamic_prediction_len = predicted_dynamic_arousal.shape[0]//len(audio_filenames)
    for idx in tqdm(range(len(audio_filenames)), desc='Emotion Features'):
        static_arousal_path = os.path.join(static_arousal_dir, 'static_arousal_' + audio_filenames[idx] + '.npy')
        static_valence_path = os.path.join(static_valence_dir, 'static_valence_' + audio_filenames[idx] + '.npy')
        dynamic_arousal_path = os.path.join(dynamic_arousal_dir, 'dynamic_arousal_' + audio_filenames[idx] + '.npy')
        dynamic_valence_path = os.path.join(dynamic_valence_dir, 'dynamic_valence_' + audio_filenames[idx] + '.npy')

        if os.path.exists(static_arousal_path) and os.path.exists(static_valence_path) and os.path.exists(dynamic_arousal_path) and os.path.exists(dynamic_valence_path):
            continue
        else:
            if not os.path.exists(static_arousal_path):
                np.save(static_arousal_path, predicted_static_arousal[idx])
            if not os.path.exists(static_valence_path):
                np.save(static_valence_path, predicted_static_valence[idx])
            if not os.path.exists(dynamic_arousal_path):
                np.save(dynamic_arousal_path, predicted_dynamic_arousal[idx*single_dynamic_prediction_len : (idx+1)*single_dynamic_prediction_len])
            if not os.path.exists(dynamic_valence_path):
                np.save(dynamic_valence_path, predicted_dynamic_valence[idx*single_dynamic_prediction_len : (idx+1)*single_dynamic_prediction_len])

    print('static arousal of {} saved at {}'.format(AUDIO_DIR, static_arousal_dir))
    print('static valence of {} saved at {}'.format(AUDIO_DIR, static_valence_dir))
    print('dynamic arousal of {} saved at {}'.format(AUDIO_DIR, dynamic_arousal_dir))
    print('dynamic valence of {} saved at {}'.format(AUDIO_DIR, dynamic_valence_dir))

def extract_all_features(open_opensmile):
    extract_chord_features()
    extract_rhythm_features()
    extract_timbre_features()
    extract_emotion_features(open_opensmile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--extract_opensmile_features', help='use opensmile to extract features or not', type=bool, default=False)
    args = parser.parse_args()

    extract_all_features(args.extract_opensmile_features)

    