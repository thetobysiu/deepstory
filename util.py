# SIU KING WAI
import re
import os
import copy
import spacy
import scipy
import skvideo.io as sio
import ffmpeg
import numpy as np

from unidecode import unidecode
from skimage import transform as tf
from modules.sda import tempdir


def quote_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "“" or token.text == "”":
            doc[token.i + 1].is_sent_start = True
    return doc


nlp = spacy.load('en_core_web_sm')
nlp_no_comma = copy.deepcopy(nlp)
nlp.add_pipe(quote_boundaries, before="parser")
sentencizer = nlp.create_pipe("sentencizer")
sentencizer.punct_chars.add(',')
sentencizer_no_comma = nlp_no_comma.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, first=True)
nlp_no_comma.add_pipe(sentencizer_no_comma, first=True)


def normalize_text(text):
    """Normalize text so that some punctuations that indicate pauses will be replaced as commas"""
    replace_list = [
        [r'(\w)’(\w)', r"\1'\2"],  # fix apostrophe for content from books
        [r'\(|\)|:|;| “|(\s*-+\s+)|(\s+-+\s*)|\s*-{2,}\s*', ', '],
        [r'\s*,[^\w]*,\s*', ', '],  # capture multiple commas
        [r'\s*,\s*', ', '],  # format commas
        [r'\.,', '.'],
        [r'[‘’“”]', '']  # strip quote
    ]
    for regex, replacement in replace_list:
        text = re.sub(regex, replacement, text)
    text = re.sub(r' +', ' ', text)
    text = unidecode(text)  # Get rid of the accented characters
    return text


def separate(text, n_gram, comma, max_len=30):
    _nlp = nlp if comma else nlp_no_comma
    lines = []
    line = ''
    counter = 0
    for sent in _nlp(text).sents:
        if sent.text:
            if counter == 0:
                line = sent.text
            else:
                line = f'{line} {sent.text}'
            counter += 1

            if counter == n_gram:
                lines.append(_nlp(line))
                line = ''
                counter = 0

    # for remaining sentences
    if line:
        lines.append(_nlp(line))

    return lines


# code from sda
def save_video(video, audio, path, fs, overwrite=True, experimental_ffmpeg=False, scale=None):
    if not os.path.isabs(path):
        path = os.getcwd() + "/" + path

    with tempdir() as dirpath:
        # Save the video file
        writer = sio.FFmpegWriter(dirpath + "/tmp.avi",
                                  inputdict={'-r': str(25) + "/1", },
                                  outputdict={'-r': str(25) + "/1", }
                                  )
        for i in range(video.shape[0]):
            frame = np.rollaxis(video[i, :, :, :], 0, 3)

            if scale is not None:
                frame = tf.rescale(frame, scale, anti_aliasing=True, multichannel=True, mode='reflect')

            writer.writeFrame(frame)
        writer.close()

        # Save the audio file
        scipy.io.wavfile.write(dirpath + "/tmp.wav", fs, audio)

        in1 = ffmpeg.input(dirpath + "/tmp.avi")
        in2 = ffmpeg.input(dirpath + "/tmp.wav")
        if experimental_ffmpeg:
            out = ffmpeg.output(in1['v'], in2['a'], path, strict='-2', loglevel="panic")
        else:
            out = ffmpeg.output(in1['v'], in2['a'], path, loglevel="panic")

        if overwrite:
            out = out.overwrite_output()
        out.run()
