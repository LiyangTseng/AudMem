import os
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matlab.engine
from PMEmo.features import extract_all_wav_feature, extract_frame_feature, process_dynamic_feature
import arff
import pickle
from sklearn.preprocessing import StandardScaler

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
        chroma_path = os.path.join(chroma_dir, 'chroma_'+ audio_file[:16] + '.npy') 
        tonnetz_path =  os.path.join(tonnetz_dir, 'tonnetz_'+ audio_file[:16] + '.npy')
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
    
    ''' extract rhythm pattern (FA) from MA Toolbox '''
    
    fp_dir = 'features/rhythms/fluctuation_pattern'
    if not os.path.exists(fp_dir):
        os.makedirs(fp_dir)

    eng = matlab.engine.start_matlab()
    for audio_file in tqdm(os.listdir(AUDIO_DIR), desc='Rhythm Features', leave=True):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        fp_path = os.path.join(fp_dir, 'fp_'+ audio_file[:16] + '.npy')
        if os.path.exists(fp_path):
            continue

        wav = eng.audioread(audio_path)

        p = {'sequence': {'length': 512.0, 'hopsize': 256.0, 'windowfunction': 'boxcar'}, 
                'fs': 11025.0, 'fft_hopsize': 128.0, 'visu': 0}
        sone = eng.ma_sone(wav, p)
        fp = eng.ma_fp(sone,p)
        fp = np.array(fp._data) # convert mlarray.double to numpy array 
        
        np.save(fp_path, fp)

    print('fluctuation pattern features of {} saved at {}'.format(AUDIO_DIR, fp_dir))
        
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
        mfcc_path = os.path.join(mfcc_dir, 'mfcc_' + audio_file[:16] + '.npy')
        mfcc_delta_path = os.path.join(mfcc_delta_dir, 'mfccDelta_' + audio_file[:16] + '.npy')
        mfcc_delta2_path = os.path.join(mfcc_delta2_dir, 'mfccDelta2_' + audio_file[:16] + '.npy')
        spectral_centroid_path = os.path.join(spectral_centroid_dir, 'cent_' + audio_file[:16] + '.npy')
        spectral_bandwidth_path = os.path.join(spectral_bandwidth_dir, 'specBw_' + audio_file[:16] + '.npy') 
        spectral_contrast_path = os.path.join(spectral_contrast_dir, 'contrast_' + audio_file[:16] + '.npy')
        spectral_flatness_path = os.path.join(spectral_flatness_dir, 'flatness_' + audio_file[:16] + '.npy')
        spectral_rolloff_path = os.path.join(spectral_rolloff_dir, 'rolloff_' + audio_file[:16] + '.npy')

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

def extract_emotion_features(args):

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

    if args.extract_features:
        extract_opensmile_features()


    ''' predict static arousal and valence using PMEmo '''
    with open('features/emotions/tmp/static_features.arff') as f:
        arrf_data = arff.load(f)
    static_features = np.array(arrf_data['data'])[:, 1:-1]
    scaler = StandardScaler().fit(static_features)
    scaled_dynamic_features = scaler.transform(static_features)

    audio_filenames = [data[0][:-4] for data in arrf_data['data']]

    # load saved model ref: https://stackoverflow.com/questions/32700797/saving-a-cross-validation-trained-model-in-scikit
    with open('features/emotions/tmp/svr_linear_static_arousal.pkl' , 'rb') as fid:
        static_arousal_model = pickle.load(fid)  
    with open('features/emotions/tmp/svr_linear_static_valence.pkl' , 'rb') as fid:
        static_valence_model = pickle.load(fid)

    predicted_static_arousal = static_arousal_model.predict(scaled_dynamic_features)
    predicted_static_valence = static_valence_model.predict(scaled_dynamic_features)

    ''' predict dynamic arousal and valence using PMEmo '''
    csv_data = pd.read_csv('features/emotions/tmp/dynamic_features.csv')
    dynamic_features = csv_data[csv_data.columns[2:262]]
    scaler = StandardScaler().fit(dynamic_features)
    scaled_dynamic_features = scaler.transform(dynamic_features)
    
    with open('features/emotions/tmp/svr_rbf_dynamic_arousal.pkl' , 'rb') as fid:
        dynamic_arousal_model = pickle.load(fid)
    with open('features/emotions/tmp/svr_rbf_dynamic_valence.pkl' , 'rb') as fid:
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
        static_arousal_path = os.path.join(static_arousal_dir, 'staticArousal_' + audio_filenames[idx] + '.npy')
        static_valence_path = os.path.join(static_valence_dir, 'staticValence_' + audio_filenames[idx] + '.npy')
        dynamic_arousal_path = os.path.join(dynamic_arousal_dir, 'dynamicArousal_' + audio_filenames[idx] + '.npy')
        dynamic_valence_path = os.path.join(dynamic_valence_dir, 'dynamicValence_' + audio_filenames[idx] + '.npy')

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--extract_features', help='use opensmile to extract features or not', type=bool, default=False)
    args = parser.parse_args()


    extract_chord_features()
    extract_rhythm_features()
    extract_timbre_features()
    extract_emotion_features(args)

    