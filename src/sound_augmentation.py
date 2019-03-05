import np
import random
import os.path
import librosa
from pysndfx import AudioEffectsChain
import sox 


def augment_sound(input_audio):
    #TODO: Add aachen room impulse response:
    # https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
    with_shift = shift(input_audio)
    with_bg_noise = add_background_noise(with_shift)
    #Very slow: (Might not be used in future...)
    with_reverb = add_reverb(with_bg_noise)
    return with_reverb

def add_background_noise(input_audio):
    #TODO: Gaussian distribution num = min(10, max(0, random.gauss(3, 4)))
    fn = os.path.join(os.path.dirname(__file__), '../_background_noise_/')
    background_audio_files = os.listdir(fn)
    background_audio_path = background_audio_files[random.randint(0, len(background_audio_files)-1)]
    bg_noise, sr = librosa.load(fn + background_audio_path, sr=16000)
    random_noise_level = random.uniform(0, 0.1)
    result = (1 - random_noise_level) * input_audio + random_noise_level * bg_noise[:len(input_audio)]
    return result

def shift(input_audio):
    # Percentage to be shifted: 
    timeshift_fac = 0.3 *2*(np.random.uniform()-0.5)  # up to 20% of length
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
    .reverb(reverberance=50 * reverb_fac,
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