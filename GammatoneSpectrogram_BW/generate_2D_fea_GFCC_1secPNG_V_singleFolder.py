from __future__ import print_function
from python_speech_features import mfcc
from python_speech_features import logfbank
import librosa
import librosa.display
import numpy as np
from os import listdir, remove
from os.path import join, isdir
import natsort
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('agg')  # no images shown
import warnings
warnings.filterwarnings("ignore")

import gammatone.gtgram
import gammatone.fftweight
from plot_UrbanSound8K import render_GFCC_from_sig

'''
Number of classes = 10
class_labels=[
'air_conditioner',
'car_horn',
'children_playing',
'dog_bark',
'drilling',
'engine_idling',
'gun_shot',
'jackhammer',
'siren',
'street_music'
]

['./air_conditioner',
 './car_horn',
 './children_playing',
 './dog_bark',
 './drilling',
 './engine_idling',
 './gun_shot',
 './jackhammer',
 './siren',
 './street_music'] 

'''
def remove_csv_files(root_folder):
    main_dir = root_folder  # folder for images, the default is current local folder
    # folder_names = [join(main_dir, x) for x in listdir(main_dir) if is_image_file(x)]
    folder_names = [join(main_dir, x) for x in listdir(main_dir) if
                    isdir(x) and not x.startswith('.') and not x.startswith('_')]
    folder_names = natsort.natsorted(folder_names, reverse=False)

    for folder_index in range(len(folder_names)):
        print('\nProcessing... {:.0f}%, \n   {}'.format(100 * (folder_index + 1) / len(folder_names),
                                                        folder_names[folder_index]))
        for x in listdir(folder_names[folder_index]):
            if x.endswith(".csv"):
                remove(folder_names[folder_index] + '/' + x)
                print('   removing..',x)

def remove_png_files(root_folder):
    main_dir = root_folder  # folder for images, the default is current local folder
    # folder_names = [join(main_dir, x) for x in listdir(main_dir) if is_image_file(x)]
    folder_names = [join(main_dir, x) for x in listdir(main_dir) if
                    isdir(x) and not x.startswith('.') and not x.startswith('_')]
    folder_names = natsort.natsorted(folder_names, reverse=False)

    for folder_index in range(len(folder_names)):
        print('\nProcessing... {:.0f}%, \n   {}'.format(100 * (folder_index + 1) / len(folder_names),
                                                        folder_names[folder_index]))
        for x in listdir(folder_names[folder_index]):
            if x.endswith(".png"):
                remove(folder_names[folder_index] + '/' + x)
                print('      removing..',x)

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in [".mp3", ".wav", ".flac", ".aif", ".aiff", ".ogg"]) #(, ".wma")

def zero_pad(sig, window_bits):
    """Returns a zero padded signal based on signal and necessary window_bits."""
    zero_pads = window_bits - len(sig)
    # print('zero_pads:', zero_pads)
    if np.all([zero_pads > 0, window_bits > 0, len(sig) < window_bits]):
        sig = np.pad(sig, (0, zero_pads), 'constant', constant_values=0)
    return sig


# returns list of starts and ends for each segments
def segment_audio_file(sig, start, end, sr=16000, window=2, stride_ratio=0.5, offset=-1, min_sig=0.5, show_Infor = False):
    """
        Returns a list of starts and ends for sound segments.
            Maximum processing frequency will be: sr/2 Hz;
            Window = 2 second patches
            stride_ratio=0.5
    """
    starts_ends = []
    start_sec = start
    end_sec = end
    # print('start:',start_sec,'sec =',start)
    # print('end:',end_sec,'sec =',end)

    #if (start > end):
    #    (start, end) = (end, start)
    if (start <= 0 or start > len(sig) / sr):  # start = 0
        start = 0
    else:
        start = int(start * sr)
    if (end == -1 or end > len(sig) / sr):  # end = -1
        end = len(sig)
    else:
        end = int(end * sr)

    seg_len = window * sr
    stride_step = int(seg_len * stride_ratio)
    if show_Infor:
        print(f'end: {end}')
        print(f'signal rate: {sr}')
        print(f'window (segment) length: {window} seconds ({seg_len} bits)')
        print(f'stride step: {stride_step} bits')
        print(f'offset: {offset}')

    k = 0
    if len(sig) < seg_len:
        starts_ends.append(zero_pad(sig, seg_len)) #np.concatenate((a,b))
    else:
        if (offset > 0):
            max_offset = int((1 - min_sig) * seg_len)
            offset = min(int(offset * seg_len), max_offset)
            start_orig = start
            start = max(start - offset, 0)
            offset = start_orig - start
            print(f'offset: {offset}')
            print(f'start with offset: {start}')
        # Creating multiple window_size segments from a larger than window interval
        min_bits = int(min_sig * seg_len)  # minimum singnal bits should be min_sig ratio of segment length
        # while(start+(k+1)*seg_len<=min(end,len(sig))+stride_ratio*seg_len):
        while (True):
            new_start = start + k * stride_step
            new_end = new_start + seg_len
            new_end_ori = new_end
            if new_end > len(sig):
                new_end = len(sig)
            if any([(new_end - new_start) < min_bits, (end - new_start) < min_bits]):
                break
            else:
                if new_end_ori>len(sig):
                    starts_ends.append(zero_pad(sig[new_start:new_end], seg_len))  # np.concatenate((a,b))
                else:
                    starts_ends.append(sig[new_start:new_end])
                k += 1

    return starts_ends

def seg_aud_1s_no_padding(sig, start, end, sr=16000, window=2, stride_ratio=0.5, offset=-1, min_sig=0.95, show_Infor = False):
    """
        Returns a list of starts and ends for sound segments.
            Maximum processing frequency will be: sr/2 Hz;
            Window = 1 second patches
            stride_ratio = moving_ratio = 0.5
            no_padding to the last
    """
    starts_ends = []
    start_sec = start
    end_sec = end
    # print('start:',start_sec,'sec =',start)
    # print('end:',end_sec,'sec =',end)

    #if (start > end):
    #    (start, end) = (end, start)
    if (start <= 0 or start > len(sig) / sr):  # start = 0
        start = 0
    else:
        start = int(start * sr)
    if (end == -1 or end > len(sig) / sr):  # end = -1
        end = len(sig)
    else:
        end = int(end * sr)

    seg_len = window * sr
    stride_step = int(seg_len * stride_ratio)
    if show_Infor:
        print(f'end: {end}')
        print(f'signal rate: {sr}')
        print(f'window (segment) length: {window} seconds ({seg_len} bits)')
        print(f'stride step: {stride_step} bits')
        print(f'offset: {offset}')

    k = 0
    if len(sig) <= seg_len: #not larger than 1 sec long audio file
        starts_ends.append(sig) #np.concatenate((a,b))
    else:
        if (offset > 0):
            max_offset = int((1 - min_sig) * seg_len)
            offset = min(int(offset * seg_len), max_offset)
            start_orig = start
            start = max(start - offset, 0)
            offset = start_orig - start
            print(f'offset: {offset}')
            print(f'start with offset: {start}')
        # Creating multiple window_size segments from a larger than window interval
        min_bits = int(min_sig * seg_len)  # minimum singnal bits should be min_sig ratio of segment length
        # while(start+(k+1)*seg_len<=min(end,len(sig))+stride_ratio*seg_len):
        while (True):
            new_start = start + k * stride_step
            new_end = new_start + seg_len
            if new_end >= len(sig):
                #print(new_end, len(sig), min_bits)
                if (new_end - len(sig)) < min_bits:
                    starts_ends.append(sig[len(sig)-seg_len:len(sig)])  # np.concatenate((a,b))
                break
            else:
                starts_ends.append(sig[new_start:new_end])
            k += 1

    return starts_ends

if __name__ == '__main__':
    # --- Hyperparameters ---#
    window_size = 2048  # change from 1024 to 2048
    hop_length = 512  # ---Set the hop length; at 44100 Hz, 512 samples ~= 12ms (<-- 512/44100)
    window = np.hanning(window_size)

    sample_rate_2D = 22050*2
    main_dir = './'   # folder for images, the default is current local folder

    #remove_png_files(main_dir)

    #try:
    file_count_Record = []

    overlap_Ratio = 0.7
    stride_Ratio = 1 - overlap_Ratio
    audio_filenames = [join(main_dir, x) for x in listdir(main_dir) if
                       is_audio_file(x)]
    audio_filenames = natsort.natsorted(audio_filenames, reverse=False)
    file_count_Record.append(len(audio_filenames))
    if len(audio_filenames) == 0:
        print('      Warning: no images in folder: {}'.format(folder_names[folder_index]))
    for audio_index in range(len(audio_filenames)):
        # --- Process each audio file in current categorical folder ---#
        print('      {}'.format(audio_filenames[audio_index]))
        filename = audio_filenames[audio_index]
        sig, rate = librosa.load(filename, sr=sample_rate_2D, mono=True)
        #--- Generate 1D data patches ---#

        #one_sec_segment = segment_audio_file(sig, start=0, end=-1, sr=sample_rate_2D, window=1, stride_ratio=stride_Ratio, offset=-1, min_sig=0.8, show_Infor = False)  #with padding
        one_sec_segment = seg_aud_1s_no_padding(sig, start=0, end=-1, sr=sample_rate_2D, window=1, stride_ratio=stride_Ratio,
                                         offset=-1, min_sig=0.95, show_Infor = False)  #without padding
        # --- Generate 2D spectrogram and save as PNG files ---#
        for k in range(one_sec_segment.__len__()):
            if one_sec_segment.__len__() == 1: #less or equal 1 sec long audio file
                pngFile_name = filename + '_GFCC' + '.png' #just one file
            else:
                pngFile_name = filename + '_GFCC_' + str(k + 1) + '.png'
            # Compute GFCC and return img
            #  >> duration=False: duration=len(audio)
            #  >> function=gammatone.gtgram.gtgram: slower, larger, but more accurate
            #  >> function=gammatone.fftweight.fft_gtgram: faster, lighter, but less accurate
            render_GFCC_from_sig(sig=one_sec_segment[k], sample_rate_2D=sample_rate_2D, duration=False, function=gammatone.gtgram.gtgram, pngFile_name=pngFile_name)

    '''
    except:
        #--- error and continue ---#
        print('\n !!!! Error occurred!')
        print(main_dir, audio_filenames, audio_index)
        pass
    '''
    print('----------------------------------------------------------------------')
    print('Run finished!')
