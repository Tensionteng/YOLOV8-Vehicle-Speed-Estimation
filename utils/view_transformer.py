import numpy as np
import cv2


class ViewTransformer:
    """A Class to transform camera view to real-world distance."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Initialize class with source array and target array."""
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: tuple) -> tuple:
        """transform source point to target point and return a tuple (x,y)"""
        x, y = points
        reshaped_points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return (transformed_points[0][0][0], transformed_points[0][0][1])
