from collections import defaultdict

import cv2
import numpy as np
from utils import view_transformer

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator:
    """A class to estimation speed of objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the speed-estimator class with default values for Visual, Image, track and speed parameters."""

        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False

        # Region information
        self.region_thickness = 3

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.line_thickness = 2
        self.trk_history = defaultdict(list)

        # Speed estimator information
        self.dist_data = {}
        self.track_points_history = defaultdict(list)

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        source_region,
        target_region,
        fps,
        names,
        view_img=False,
        line_thickness=2,
        region_thickness=5,
    ):
        """
        Configures the speed estimation and display parameters.

        Args:
            source_region: Video pixel region.
            target_region: Real-world region(unit: meters).
            fps: Video fps.
            names (dict): object detection classes names
            view_img (bool): Flag indicating frame display
            line_thickness (int): Line thickness for bounding boxes.
            region_thickness (int): Speed estimation region thickness
        """

        self.source_region = source_region
        self.target_region = target_region
        self.fps = fps
        self.names = names
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.region_thickness = region_thickness
        self.view_transformer = view_transformer.ViewTransformer(
            source=self.source_region, target=self.target_region
        )

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def store_track_info(self, track_id, box):
        """
        Store track data.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
        """
        track = self.trk_history[track_id]
        bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
        track.append(bbox_center)

        if len(track) > 30:
            track.pop(0)

        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        Plot track and bounding box.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
            cls (str): object class name
            track (list): tracking history for tracks path drawing
        """
        speed_label = (
            f"{int(self.dist_data[track_id])}km/h"
            if track_id in self.dist_data
            else self.names[int(cls)]
        )
        bbox_color = (
            colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)
        )

        self.annotator.box_label(box, speed_label, bbox_color)

        cv2.polylines(
            self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1
        )
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Calculation of object speed
        Args:
            trk_id (int): object track id.
            track (list): tracking history for tracks path drawing
        """

        # check if track object is in the source region
        if (
            cv2.pointPolygonTest(
                self.source_region, (track[-1][0], track[-1][1]), measureDist=False
            )
            < 0
        ):
            return

        if len(self.track_points_history[trk_id]) > 1:
            distance = abs(
                self.track_points_history[trk_id][-1][1]
                - self.track_points_history[trk_id][0][1]
            )
            time_diff = len(self.track_points_history[trk_id]) / self.fps
            speed = distance / time_diff * 3.6
            self.dist_data[trk_id] = speed

        point = self.view_transformer.transform_points(track[-1])
        self.track_points_history[trk_id].append(point)

    def estimate_speed(self, im0, tracks):
        """
        Calculate object based on tracking data
        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
            return
        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)
        self.annotator.draw_region(
            reg_pts=self.source_region,
            color=(0, 255, 0),
            thickness=self.region_thickness,
        )

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track)

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Vehicle Speed Estimation", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    SpeedEstimator()
