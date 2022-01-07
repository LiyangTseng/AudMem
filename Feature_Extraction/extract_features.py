import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from features import extract_all_wav_feature, extract_frame_feature, process_dynamic_feature
import arff
import pickle
from sklearn.preprocessing import StandardScaler
'''
    this file is the modified version of the one in Feature_Extraction/extract_features.py
'''

def normalize_features(np_array):
    ''' normalize numpy array to [-1, 1] '''
    # ref: https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range 
    return 2.*(np_array - np.min(np_array))/np.ptp(np_array)-1

def extract_chord_features(audio_dir):

    ''' extract choragram and tonnetz from LibROSA'''

    chroma_dir = '{}/chords/chroma'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(chroma_dir):
        os.makedirs(chroma_dir)
    tonnetz_dir = '{}/chords/tonnetz'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(tonnetz_dir):
        os.makedirs(tonnetz_dir)

    for audio_file in tqdm(os.listdir(audio_dir), desc='Chord Features', leave=True):
        audio_path = os.path.join(audio_dir, audio_file)
        chroma_path = os.path.join(chroma_dir, 'chroma_'+ audio_file.replace(".wav", ".npy")) 
        tonnetz_path =  os.path.join(tonnetz_dir, 'tonnetz_'+ audio_file.replace(".wav", ".npy"))
        if os.path.exists(chroma_path) and os.path.exists(tonnetz_path):
            continue
        else:
            y, sr = librosa.load(audio_path)
            if not os.path.exists(chroma_path):
                chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
                chroma_cqt = normalize_features(chroma_cqt)
                np.save(chroma_path, chroma_cqt)
            if not os.path.exists(tonnetz_path):
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz = normalize_features(tonnetz)
                np.save(tonnetz_path, tonnetz)

    print('chroma features of {} saved at {}'.format(audio_dir, chroma_dir))
    print('tonnetz features of {} saved at {}'.format(audio_dir, tonnetz_dir))

def extract_rhythm_features(audio_dir):
    
    ''' extract tempogram from libROSA '''
    
    tempogram_dir = '{}/rhythms/tempogram'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(tempogram_dir):
        os.makedirs(tempogram_dir)


    for audio_file in tqdm(os.listdir(audio_dir), desc='Rhythm Features', leave=True):
        audio_path = os.path.join(audio_dir, audio_file)
        tempogram_path = os.path.join(tempogram_dir, 'tempogram_'+ audio_file.replace(".wav", ".npy"))
        if os.path.exists(tempogram_path):
            continue

        y, sr = librosa.load(audio_path)
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        tempogram = normalize_features(tempogram)
        np.save(tempogram_path, tempogram)

    print('tempogram features of {} saved at {}'.format(audio_dir, tempogram_dir))
        
def extract_timbre_features(audio_dir):

    ''' extract MFCC and features related to spectrum shape from LibROSA'''
    
    mfcc_dir = '{}/timbre/mfcc'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(mfcc_dir):
        os.makedirs(mfcc_dir)
    mfcc_delta_dir = '{}/timbre/mfcc_delta'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(mfcc_delta_dir):
        os.makedirs(mfcc_delta_dir)
    mfcc_delta2_dir = '{}/timbre/mfcc_delta2'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(mfcc_delta2_dir):
        os.makedirs(mfcc_delta2_dir)
    spectral_centroid_dir = '{}/timbre/spectral_centroid'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(spectral_centroid_dir):
        os.makedirs(spectral_centroid_dir)
    spectral_bandwidth_dir = '{}/timbre/spectral_bandwidth'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(spectral_bandwidth_dir):
        os.makedirs(spectral_bandwidth_dir)
    spectral_contrast_dir = '{}/timbre/spectral_contrast'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(spectral_contrast_dir):
        os.makedirs(spectral_contrast_dir)
    spectral_flatness_dir = '{}/timbre/spectral_flatness'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(spectral_flatness_dir):
        os.makedirs(spectral_flatness_dir)
    spectral_rolloff_dir = '{}/timbre/spectral_rolloff'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(spectral_rolloff_dir):
        os.makedirs(spectral_rolloff_dir)
    melspectrogram_dir = '{}/timbre/melspectrogram'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(melspectrogram_dir):
        os.makedirs(melspectrogram_dir)

    for audio_file in tqdm(os.listdir(audio_dir), desc='Timbre Features', leave=True):
        audio_path = os.path.join(audio_dir, audio_file)
        mfcc_path = os.path.join(mfcc_dir, 'mfcc_' + audio_file.replace(".wav", ".npy"))
        mfcc_delta_path = os.path.join(mfcc_delta_dir, 'mfcc_delta_' + audio_file.replace(".wav", ".npy"))
        mfcc_delta2_path = os.path.join(mfcc_delta2_dir, 'mfcc_delta2_' + audio_file.replace(".wav", ".npy"))
        spectral_centroid_path = os.path.join(spectral_centroid_dir, 'spectral_centroid_' + audio_file.replace(".wav", ".npy"))
        spectral_bandwidth_path = os.path.join(spectral_bandwidth_dir, 'spectral_bandwidth_' + audio_file.replace(".wav", ".npy"))
        spectral_contrast_path = os.path.join(spectral_contrast_dir, 'spectral_contrast_' + audio_file.replace(".wav", ".npy"))
        spectral_flatness_path = os.path.join(spectral_flatness_dir, 'spectral_flatness_' + audio_file.replace(".wav", ".npy"))
        spectral_rolloff_path = os.path.join(spectral_rolloff_dir, 'spectral_rolloff_' + audio_file.replace(".wav", ".npy"))
        melspectrogram_path = os.path.join(melspectrogram_dir, 'melspectrogram_' + audio_file.replace(".wav", ".npy"))

        if os.path.exists(mfcc_path) and os.path.exists(mfcc_delta_path) and os.path.exists(mfcc_delta2_path) and \
            os.path.exists(spectral_centroid_path) and os.path.exists(spectral_bandwidth_path) and os.path.exists(spectral_contrast_path) and \
            os.path.exists(spectral_flatness_path) and os.path.exists(spectral_rolloff_path) and os.path.exists(melspectrogram_path):
            continue
        else:
            y, sr = librosa.load(audio_path)
            if not os.path.exists(mfcc_path):
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                mfcc = normalize_features(mfcc)
                np.save(mfcc_path, mfcc)
            if not os.path.exists(mfcc_delta_path):
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta = normalize_features(mfcc_delta)
                np.save(mfcc_delta_path, mfcc_delta)
            if not os.path.exists(mfcc_delta2_path):
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_delta2 = normalize_features(mfcc_delta2)
                np.save(mfcc_delta2_path, mfcc_delta2)
            if not os.path.exists(spectral_centroid_path):
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                cent = normalize_features(cent)
                np.save(spectral_centroid_path, cent)
            if not os.path.exists(spectral_bandwidth_path):
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                spec_bw = normalize_features(spec_bw)
                np.save(spectral_bandwidth_path, spec_bw)
            if not os.path.exists(spectral_contrast_path):
                S = np.abs(librosa.stft(y))
                contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
                contrast = normalize_features(contrast)
                np.save(spectral_contrast_path, contrast)
            if not os.path.exists(spectral_flatness_path):
                flatness = librosa.feature.spectral_flatness(y=y)
                flatness = normalize_features(flatness)
                np.save(spectral_flatness_path, flatness)
            if not os.path.exists(spectral_rolloff_path):       
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                rolloff = normalize_features(rolloff)
                np.save(spectral_rolloff_path, rolloff)
            if not os.path.exists(melspectrogram_path):       
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
                np.save(melspectrogram_path, melspectrogram)



    print('mfcc features of {} saved at {}'.format(audio_dir, mfcc_dir))
    print('mfcc_delta features of {} saved at {}'.format(audio_dir, mfcc_delta_dir))
    print('mfcc_delta2 features of {} saved at {}'.format(audio_dir, mfcc_delta2_dir))
    print('spectral_centroid features of {} saved at {}'.format(audio_dir, spectral_centroid_dir))
    print('spectral_bandwidth features of {} saved at {}'.format(audio_dir, spectral_bandwidth_dir))
    print('spectral_contrast features of {} saved at {}'.format(audio_dir, spectral_contrast_dir))
    print('spectral_flatness features of {} saved at {}'.format(audio_dir, spectral_flatness_dir))
    print('spectral_rolloff features of {} saved at {}'.format(audio_dir, spectral_rolloff_dir))
    print('melspectrogram features of {} saved at {}'.format(audio_dir, melspectrogram_dir))

def extract_emotion_features(audio_dir, open_opensmile = False):

    ''' extract predicted valence, arousal from PMEmo '''
    
    def extract_opensmile_features():
        # extract features for PMEmo 
        wavdir = audio_dir
        opensmiledir = 'opensmile'
        tmp_feature_folder = '{}/emotions/tmp'.format(audio_dir.replace("raw_audios", "features"))
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
    with open('{}/emotions/tmp/static_features.arff'.format(audio_dir.replace("raw_audios", "features"))) as f:
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
    csv_data = pd.read_csv('{}/emotions/tmp/dynamic_features.csv'.format(audio_dir.replace("raw_audios", "features")))
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
    static_arousal_dir = '{}/emotions/static_arousal'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(static_arousal_dir):
        os.makedirs(static_arousal_dir)
    static_valence_dir = '{}/emotions/static_valence'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(static_valence_dir):
        os.makedirs(static_valence_dir)
    dynamic_arousal_dir = '{}/emotions/dynamic_arousal'.format(audio_dir.replace("raw_audios", "features"))
    if not os.path.exists(dynamic_arousal_dir):
        os.makedirs(dynamic_arousal_dir)
    dynamic_valence_dir = '{}/emotions/dynamic_valence'.format(audio_dir.replace("raw_audios", "features"))
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

    print('static arousal of {} saved at {}'.format(audio_dir, static_arousal_dir))
    print('static valence of {} saved at {}'.format(audio_dir, static_valence_dir))
    print('dynamic arousal of {} saved at {}'.format(audio_dir, dynamic_arousal_dir))
    print('dynamic valence of {} saved at {}'.format(audio_dir, dynamic_valence_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--extract_opensmile_features', help='use opensmile to extract features or not', type=bool, default=False)
    args = parser.parse_args()

    audio_dir_root = '../data/raw_audios'
    audio_sub_dir_list = ["original", "-5_semitones", "-4_semitones", "-3_semitones", 
                            "-2_semitones", "-1_semitones", "1_semitones", "2_semitones", "3_semitones", 
                            "4_semitones", "5_semitones"]
    for sub_dir in audio_sub_dir_list:
        audio_sub_dir = os.path.join(audio_dir_root, sub_dir)
        print("==== Processing {} ====".format(audio_sub_dir))
        extract_chord_features(audio_sub_dir)
        extract_rhythm_features(audio_sub_dir)
        extract_timbre_features(audio_sub_dir)
        # extract_emotion_features(audio_sub_dir, args.extract_opensmile_features)
        

    