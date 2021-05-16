from dataclasses import dataclass
import soundfile as sf
import torch
@dataclass
class wav_loader:
    wav_scp: str = None
    frontend: str = None
    normalize: str = None

    def __iter__(self):
        with open(self.wav_scp) as f:
            for line in f:
                key, wav_path = line.strip().split()
                feat , _ = sf.read(wav_path)
                feat = torch.from_numpy(feat).unsqueeze(0)
                # if frontend is not None:
                #     feat, feat_lengths = self.frontend(feat, speech_lengths)
                # if normalize is not None:
                #     feat, feat_lengths = self.normalize(feat, feat_lengths)

                yield [key], {'speech': feat, 'speech_lengths': feat.shape[1]}


