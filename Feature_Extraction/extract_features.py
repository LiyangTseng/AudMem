import os
import numpy as np
import librosa
from tqdm import tqdm

AUDIO_DIR = '../Data_Collection/Audios/clips'


def extract_chord_features():

    ''' extract choragram and tonnetz from LibROSA'''

    chormaDir = 'features/chords/chorma'
    if not os.path.exists(chormaDir):
        os.makedirs(chormaDir)
    tonnetzDir = 'features/chords/tonnetz'
    if not os.path.exists(tonnetzDir):
        os.makedirs(tonnetzDir)

    for audioFile in tqdm(os.listdir(AUDIO_DIR)):
        audioPath = os.path.join(AUDIO_DIR, audioFile)
        chormaFile = 'chroma_'+ audioFile[:16]
        tonnetzFile = 'tonnetz_'+ audioFile[:16]
        y, sr = librosa.load(audioPath)
        chroma_cqt = librosa.feature.chroma_cens(y=y, sr=sr)
        np.save(os.path.join(chormaDir, chormaFile), chroma_cqt)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        np.save(os.path.join(tonnetzDir, tonnetzFile), tonnetz)

    print('chroma extracted from {} and stored at {}'.format(AUDIO_DIR, chormaDir))
    print('tonnetz extracted from {} and stored at {}'.format(AUDIO_DIR, tonnetzDir))

def extract_rhythm_features():
    ''' extract rhythm pattern (FA) from MA Toolbox '''
    pass

def extract_timbre_features():

    ''' extract MFCC and features related to spectrum shape from LibROSA'''
    
    mfccDir = 'features/timbre/mfcc'
    if not os.path.exists(mfccDir):
        os.makedirs(mfccDir)
    mfccDeltaDir = 'features/timbre/mfcc_delta'
    if not os.path.exists(mfccDeltaDir):
        os.makedirs(mfccDeltaDir)
    mfccDelta2Dir = 'features/timbre/mfcc_delta2'
    if not os.path.exists(mfccDelta2Dir):
        os.makedirs(mfccDelta2Dir)
    spectralCentroidDir = 'features/timbre/spectral_centroid'
    if not os.path.exists(spectralCentroidDir):
        os.makedirs(spectralCentroidDir)
    spectralBandwidthDir = 'features/timbre/spectral_bandwidth'
    if not os.path.exists(spectralBandwidthDir):
        os.makedirs(spectralBandwidthDir)
    spectralContrastDir = 'features/timbre/spectral_contrast'
    if not os.path.exists(spectralContrastDir):
        os.makedirs(spectralContrastDir)
    spectralFlatnessDir = 'features/timbre/spectral_flatness'
    if not os.path.exists(spectralFlatnessDir):
        os.makedirs(spectralFlatnessDir)
    spectralRolloffDir = 'features/timbre/spectral_rolloff'
    if not os.path.exists(spectralRolloffDir):
        os.makedirs(spectralRolloffDir)
     
    for audioFile in tqdm(os.listdir(AUDIO_DIR)):
        audioPath = os.path.join(AUDIO_DIR, audioFile)
        mfccFile = 'mfcc_' + audioFile[:16]
        mfccDeltaFile = 'mfccDelta_' + audioFile[:16]
        mfccDelta2File = 'mfccDelta2_' + audioFile[:16]
        spectralCentroidFile = 'spectralCentroid' + audioFile[:16]
        spectralBandwidthFile = 'spectralBandwidth' + audioFile[:16]
        spectralContrastFile = 'spectralContrast' + audioFile[:16]
        spectralFlatnessFile = 'spectralFlatness' + audioFile[:16]
        spectralRolloffFile = 'spectralRolloff' + audioFile[:16]
        y, sr = librosa.load(audioPath)
        mfcc = librosa.feature.chroma_cens(y=y, sr=sr)
        np.save(os.path.join(mfccDir, mfccFile), mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        np.save(os.path.join(mfccDeltaDir, mfccDeltaFile), mfcc_delta)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        np.save(os.path.join(mfccDelta2Dir, mfccDelta2File), mfcc_delta2)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        np.save(os.path.join(spectralCentroidDir, spectralContrastFile), cent)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        np.save(os.path.join(spectralBandwidthDir, spectralBandwidthFile), spec_bw)
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        np.save(os.path.join(spectralCentroidDir, spectralContrastFile), contrast)
        flatness = librosa.feature.spectral_flatness(y=y)
        np.save(os.path.join(spectralFlatnessDir, spectralFlatnessFile), flatness)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        np.save(os.path.join(spectralRolloffDir, spectralRolloffFile), rolloff)
        

    print('mfcc extracted from {} and stored at {}'.format(AUDIO_DIR, mfccDir))
    print('mfcc_delta extracted from {} and stored at {}'.format(AUDIO_DIR, mfccDeltaDir))
    print('mfcc_delta2 extracted from {} and stored at {}'.format(AUDIO_DIR, mfccDelta2Dir))
    print('spectral_centroid extracted from {} and stored at {}'.format(AUDIO_DIR, spectralCentroidDir))
    print('spectral_bandwidth extracted from {} and stored at {}'.format(AUDIO_DIR, spectralBandwidthDir))
    print('spectral_contrast extracted from {} and stored at {}'.format(AUDIO_DIR, spectralContrastDir))
    print('spectral_flatness extracted from {} and stored at {}'.format(AUDIO_DIR, spectralFlatnessDir))
    print('spectral_rollof extracted from {} and stored at {}'.format(AUDIO_DIR, spectralRolloffDir))

def extract_emotion_features():
    ''' extract predicted valence, arousal from PMEmo '''
    pass

if __name__ == '__main__':
    # extract_chord_features()
    # extract_timbre_features()