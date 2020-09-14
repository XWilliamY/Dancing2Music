import os
import numpy as np
import pickle
import random
import preprocess as p
from shutil import copyfile
import sys
import librosa

# infile: .wav
# outdir

infile = sys.argv[1]
outdir = sys.argv[2]

if not os.path.exists(outdir):
  os.mkdir(outdir)

outfile = os.path.join(outdir, 'style.npy')
p.preprocess(infile, outfile)


y, sr = librosa.load(infile) 
onset_env = librosa.onset.onset_strength(y, sr=sr,aggregate=np.median)
times = librosa.frames_to_time(np.arange(len(onset_env)),sr=sr, hop_length=512)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,sr=sr)
np.save('{}/beats'.format(outdir), times[beats])  
