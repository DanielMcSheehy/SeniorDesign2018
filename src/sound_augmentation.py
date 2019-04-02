import np
import random
import os.path
import librosa
from pysndfx import AudioEffectsChain
import sox 


def augment_sound(input_audio, background_audio):
    #TODO: Add aachen room impulse response:
    # https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
    #with_shift = shift(input_audio)
    with_shift = input_audio
    with_bg_noise = add_background_noise(with_shift, background_audio)
    #Very slow: (Might not be used in future..)
    # with_reverb = add_reverb(with_bg_noise)
    with_reverb = with_bg_noise
    return with_reverb

def load_background_audio():
    fn = os.path.join(os.path.dirname(__file__), '../_background_noise_/')
    background_audio_files = os.listdir(fn)
    background_audio = []
    for background_audio_path in background_audio_files:
        bg_noise, sr = librosa.load(fn + background_audio_path, sr=16000)
        background_audio.append(bg_noise)
    return background_audio

def add_background_noise(input_audio, background_audio):
    #TODO: Gaussian distribution num = min(10, max(0, random.gauss(3, 4)))
    add_background_noise = True if random.uniform(0, 1) > 0.2 else False # 80% of audio is augmented
    if add_background_noise:
        chosen_background_audio = background_audio[random.randint(0, len(background_audio)-1)]
        random_noise_level = random.uniform(0, 0.1)
        result = (1 - random_noise_level) * input_audio + random_noise_level * chosen_background_audio[:len(input_audio)]
    else:   
        result = input_audio
    return result

def shift(input_audio):
    # Percentage to be shifted: 
    timeshift_fac = np.random.uniform(0, 0.10) # up to 10% of length
    start = int(input_audio.shape[0] * timeshift_fac)
    if (start > 0):
        result = np.pad(input_audio,(start,0),mode='constant')[0:input_audio.shape[0]]
    else:
        result = np.pad(input_audio,(0,-start),mode='constant')[0:input_audio.shape[0]]
    return result

def add_reverb(input_audio):
    reverb_fac = random.uniform(0, 1)
    fx = (
    AudioEffectsChain()
    .reverb( 
            reverberance=50 * reverb_fac,
            hf_damping=50 * reverb_fac,
            room_scale=100,
            stereo_depth=100,
            pre_delay=20,
            wet_gain=0,
            wet_only=False)
    )
    return fx(input_audio)

def add_room_impulse_response(input_audio):
    # eng = matlab.engine.start_matlab()
    print('whatever')