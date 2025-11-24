import numpy as np


class DistanceEstimator:
    def __init__(self, camera_height=1.5):
        self.camera_height = camera_height  # Meters (approx eye level)
        self.focal_length = 650  # Tuned for standard webcams/phones

        # Horizon adjustment: When looking down, horizon moves UP
        # 0 = Center of image. Positive = Shifted Up.
        self.horizon_offset = 100

    def calibrate(self, adjustment):
        """Adjust horizon based on user input or auto-calibration"""
        self.horizon_offset += adjustment

    def estimate(self, bbox, frame_height):
        _, _, _, y2 = bbox  # Bottom of the box (feet of the object)

        # Safety clamp
        if y2 >= frame_height:
            y2 = frame_height - 1

        # Calculate logical horizon based on camera tilt
        logical_horizon = (frame_height / 2) - self.horizon_offset

        # Pixels from horizon to object base
        pixel_offset = y2 - logical_horizon

        if pixel_offset <= 0:
            return 99.9  # Object is above horizon (floating or far away)

        # Pin-hole camera model
        dist = (self.camera_height * self.focal_length) / pixel_offset

        # Clamp output to realistic walking range (0.5m to 10m)
        return float(np.clip(dist, 0.4, 10.0))
