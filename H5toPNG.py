
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import fftconvolve
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import re
import h5py
import cv2
import os
import pandas as pd
import scipy
from scipy import constants
from scipy import interpolate
from scipy import ndimage
from scipy import stats
import time
import warnings
import glob
import cv2
import os
import numpy as np
from skimage import io as sio
import skimage
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Video:
    def __init__(self, path):
        self.relpath = os.path.relpath(path)
        self.abspath = os.path.abspath(path)
        self.filenames = [os.path.basename(path)]
    
    @property
    def label(self):
        return self.filename
    
    @property
    def data(self):
        with h5py.File(self.abspath, 'r') as file:
            data= file['camera']['frames'][...]
        data = np.flip(data, axis=3)
        data = np.right_shift(data, 6) # Converts to 10-bit data
        return data
        
    @property
    def frames(self):
        with h5py.File(self.abspath, 'r') as file:
            return file['camera']['frames'].shape[0]
    
    @property
    def rows(self):
        with h5py.File(self.abspath, 'r') as file:
            return file['camera']['frames'].shape[1]
    
    @property
    def cols(self):
        with h5py.File(self.abspath, 'r') as file:
            return file['camera']['frames'].shape[2]
    
    @property
    def chs(self):
        with h5py.File(self.abspath, 'r') as file:
            return file['camera']['frames'].shape[3]
        
    @property
    def shape(self):
        return (self.frames, self.rows, self.cols, self.chs)
        
    @property
    def exposure(self):
        with h5py.File(self.abspath, 'r') as file:
            exposure = file['camera']['integration-time']
        exposure, = set(exposure)
        return exposure
    
    @property
    def exposure_us(self):
        return self.exposure
    
    @property
    def exposure_ms(self):
        return self.exposure/1000       


class VideoClip:
    def __init__(self, path):
        self.relpath = os.path.relpath(path)
        self.abspath = os.path.abspath(path)
        filepaths = sorted(glob.glob(os.path.join(path, '*.h5')))
        self.filenames = [os.path.basename(filepath) for filepath in filepaths]    
        
        self._videos = [Video(os.path.join(self.abspath, filename)) for filename in self.filenames]
    
    @property
    def label(self):
        return os.path.basename(os.path.normpath(self.abspath))
    
    @property
    def data(self):
        data_perfile = [video.data for video in self._videos]
        data_allfiles = np.concatenate(data_perfile, axis=0)
        return data_allfiles
    
    @property
    def frames(self):
        frames_perfile = [video.frames for video in self._videos]
        frames_allfiles = sum(frames_perfile)
        return frames_allfiles
    
    @property
    def rows(self):
        rows_perfile = [video.rows for video in self._videos]
        rows_allfiles, = set(rows_perfile)
        return rows_allfiles
    
    @property
    def cols(self):
        cols_perfile = [video.cols for video in self._videos]
        cols_allfiles, = set(cols_perfile)
        return cols_allfiles
    
    @property
    def chs(self):
        chs_perfile = [video.chs for video in self._videos]
        chs_allfiles, = set(chs_perfile)
        return chs_allfiles
    
    @property
    def shape(self):
        return (self.frames, self.rows, self.cols, self.chs)
    
    @property
    def exposure(self):
        exposure_perfile = [video.exposure for video in self._videos]
        exposure_allfiles, = set(exposure_perfile)
        return exposure_allfiles
    
    @property
    def exposure_us(self):
        exposure_us_perfile = [video.exposure_us for video in self._videos]
        exposure_us_allfiles, = set(exposure_us_perfile)
        return exposure_us_allfiles
    
    @property
    def exposure_ms(self):
        exposure_ms_perfile = [video.exposure_ms for video in self._videos]
        exposure_ms_allfiles, = set(exposure_ms_perfile)
        return exposure_ms_allfiles


beta = 0.488
def correct_charge_sharing(image, beta):
    alpha = 1 - beta
    kernel = [1/alpha, -beta/alpha]
    return ndimage.convolve1d(image, kernel, axis=1)
def demosaicing_CFA_Bayer_bilinear(img):
  kernel = np.array(
          [[0, 1, 0],
            [1, 4, 1],
            [0, 1, 0]]) / 4
  R = scipy.ndimage.convolve(img[:,:,2], kernel)
  B = scipy.ndimage.convolve(img[:,:,0], kernel)
  G = scipy.ndimage.convolve(img[:,:,1], kernel)
  checkboard = np.dstack((B,G,R))
  return checkboard
def mosaicing_CFA_Bayer_1(img):
  x = np.zeros((img.shape),dtype=int)
  x[1::2,::2,:] = 1
  x[::2,1::2,:] = 1
  mosaic = img * x
  return mosaic

def mosaicing_CFA_Bayer_2(img):
  x = np.ones((img.shape),dtype=int)
  x[1::2,::2,:] = 0
  x[::2,1::2,:] = 0
  mosaic = img * x
  return mosaic
videos = []
directory_path = './'
if not os.path.isdir(directory_path):
    print(f'Error: The directory at {directory_path} does not exist.')
else:
    all_files = os.listdir(directory_path)
    h5_files = [file for file in all_files if file.lower().endswith('.h5')]

video = Video(h5_files[0]).data[0,:,:,:]
video = correct_charge_sharing(video, beta)
video_VIS = (mosaicing_CFA_Bayer_1(video)/1024)*255
video_NIR = (mosaicing_CFA_Bayer_2(video)/1024)*255
cv2.imwrite("VIS.png",video_VIS.astype(np.uint8)) 
cv2.imwrite("NIR.png",video_NIR.astype(np.uint8)) 
