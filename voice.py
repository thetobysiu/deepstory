# SIU KING WAI SM4701 Deepstory
import numpy as np
import torch
import glob

from util import normalize_text, normalize_audio
from modules.dctts import Text2Mel, SSRN, hp, spectrogram2wav

torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Voice:
    def __init__(self, speaker):
        self.speaker = speaker
        self.text2mel = None
        self.ssrn = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load(self):
        self.text2mel = Text2Mel(hp.vocab).to(device).eval()
        self.text2mel.load_state_dict(torch.load(glob.glob(f'data/dctts/{self.speaker}/t2m*.pth')[0])['state_dict'])
        self.ssrn = SSRN().to(device).eval()
        self.ssrn.load_state_dict(torch.load(f'data/dctts/{self.speaker}/ssrn.pth')['state_dict'])

    def close(self):
        del self.text2mel
        del self.ssrn
        torch.cuda.empty_cache()

    # referenced from original repo
    def synthesize(self, text, timeout=10000):
        with torch.no_grad():  # no grad to save memory
            normalized_text = normalize_text(text) + "E"  # text normalization, E: EOS
            L = torch.from_numpy(np.array([[hp.char2idx[char] for char in normalized_text]], np.long)).to(device)
            zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32)).to(device)
            Y = zeros

            for i in range(timeout):
                _, Y_t, A = self.text2mel(L, Y, monotonic_attention=True)
                Y = torch.cat((zeros, Y_t), -1)
                _, attention = torch.max(A[0, :, -1], 0)
                attention = attention.item()
                if L[0, attention] == hp.vocab.index('E'):  # EOS
                    break

            _, Z = self.ssrn(Y)  # batch ssrn instead?
            Z = Z.cpu().detach().numpy()

        wav = spectrogram2wav(Z[0, :, :].T)
        wav = normalize_audio(wav)
        return wav
