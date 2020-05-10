# SIU KING WAI SM4701 Deepstory
# mostly referenced from demo.py of first order model github repo, optimized loading in gpu vram
import imageio
import yaml
import torch
import torch.nn.functional as F
import numpy as np

from modules.fom import OcclusionAwareGenerator, KPDetector, DataParallelWithCallback, normalize_kp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageAnimator:
    def __init__(self):
        self.config_path = 'data/fom/vox-256.yaml'
        self.checkpoint_path = 'data/fom/vox-cpk.pth.tar'
        self.generator = None
        self.kp_detector = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load(self):
        with open(self.config_path) as f:
            config = yaml.load(f)

        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params']).to(device)

        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                      **config['model_params']['common_params']).to(device)

        checkpoint = torch.load(self.checkpoint_path)

        self.generator.load_state_dict(checkpoint['generator'])
        self.kp_detector.load_state_dict(checkpoint['kp_detector'])

        del checkpoint

        self.generator = DataParallelWithCallback(self.generator)
        self.kp_detector = DataParallelWithCallback(self.kp_detector)

        self.generator.eval()
        self.kp_detector.eval()

    def close(self):
        del self.generator
        del self.kp_detector
        torch.cuda.empty_cache()

    def animate_image(self, source_image, driving_video, relative=True, adapt_movement_scale=True):
        with torch.no_grad():
            predictions = []
            # ====================================================================================
            # adapted from original to optimize memory load in gpu instead of cpu
            source_image = imageio.imread(source_image)
            # normalize color to float 0-1
            source = torch.from_numpy(source_image[np.newaxis].astype(np.float32)).to('cuda') / 255
            del source_image
            source = source.permute(0, 3, 1, 2)
            # resize
            source = F.interpolate(source, size=(256, 256), mode='bilinear', align_corners=False)

            # modified to fit speech driven animation
            driving = torch.from_numpy(driving_video).to('cuda') / 255
            del driving_video
            driving = F.interpolate(driving, scale_factor=2, mode='bilinear', align_corners=False)
            # pad the left and right side of the scaled 128x96->256x192 to fit 256x256
            driving = F.pad(input=driving, pad=(32, 32, 0, 0, 0, 0, 0, 0), mode='constant', value=0)
            driving = driving.permute(1, 0, 2, 3).unsqueeze(0)
            # ====================================================================================
            kp_source = self.kp_detector(source)
            kp_driving_initial = self.kp_detector(driving[:, :, 0])

            for frame_idx in range(driving.shape[2]):
                driving_frame = driving[:, :, frame_idx]
                kp_driving = self.kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = self.generator(source, kp_source=kp_source, kp_driving=kp_norm)
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0])
        return np.array(predictions) * 255
