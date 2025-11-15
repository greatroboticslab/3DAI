'''Module for simulated camera class for test purpose'''

from __future__ import annotations

import numpy as np

from camera import Camera
from projector import Projector


class CameraSimulated(Camera):
    def __init__(self):        
        self.type = 'simulated'
        self._projector = None
        print(f'Simulated camera created')

    @staticmethod
    def get_available_cameras(cameras_num_to_find:int=2) -> list[Camera]:
        cameras = []
        for i in range(cameras_num_to_find):
            camera = CameraSimulated()
            camera._cam_id = i
            cameras.append(camera)
        return cameras

    def get_image(self) -> np.array:
        if self.projector is not None:
            np.random.seed(42)  # Fixed seed for repeatable noise across runs
            img = self._projector.corrected_pattern.copy()
            # Low noise for flatness (tune: 0.001 → 0 for zero-noise perfect plane)
            noise = np.random.normal(0, 0.001, img.shape)
            img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
            if self._cam_id == 1:  # Small shift for minimal disparity/flat Z
                img = np.roll(img, 25, axis=1)  # 25 pixel (tune: 0 for identical = NaN risk)
            return img
        else:
            raise ValueError()

    @property
    def projector(self) -> Projector:
        return self._projector

    @projector.setter
    def projector(self, projector: Projector):
        self._projector = projector

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, x):
        self._exposure = x

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, x):
        self._gain = x

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, x):
        self._gamma = x
