"""
Code that loads the dataset for training.
"""

import os
from typing import Optional, Union
import warnings
import ujson
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import lru_cache
import sys
import cv2
import gzip
import laspy
import io
import transfuser_utils as t_u
import gaussian_target as g_t
import random
from sklearn.utils.class_weight import compute_class_weight
from center_net import angle2class
from imgaug import augmenters as ia
import pickle
import tarfile
import re
from config import GlobalConfig
from enum import Enum


# Dealing with sequences, some frames are use again quickly
# Also loading measurement during indexing, prevents loading the same file n-times
LRU_CACHE_SIZE = 20


class SensorFolder(Enum):
    """
    Enum for the different sensor folders.
    """

    RGB = "rgb"
    RGB_AUGMENTED = "rgb_augmented"
    SEMANTICS = "semantics"
    SEMANTICS_AUGMENTED = "semantics_augmented"
    BEV_SEMANTICS = "bev_semantics"
    BEV_SEMANTICS_AUGMENTED = "bev_semantics_augmented"
    DEPTH = "depth"
    DEPTH_AUGMENTED = "depth_augmented"
    LIDAR = "lidar"
    MEASUREMENTS = "measurements"
    BOXES = "boxes"
    TRAJECTORIES = "boxes"


class CARLA_Data(Dataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        config: GlobalConfig,
        estimate_class_distributions: bool = False,
        estimate_sem_distribution: bool = False,
        shared_dict: Optional[dict] = None,
        rank: int = 0,
        validation: bool = False,
        heuristic_pruning: bool = False,
    ) -> None:

        self.data_root = "/".join(root[0].split("/")[:-1]) if len(root) != 0 else ""
        self.heuristic_pruning = heuristic_pruning

        self.config = config
        self.validation = validation

        self.data_cache = shared_dict
        self.target_speed_bins = np.array(config.target_speed_bins)
        self.angle_bins = np.array(config.angle_bins)
        self.converter = np.uint8(config.converter)
        self.bev_converter = np.uint8(config.bev_converter)

        self.route_root = []
        self.sample_start = []

        self.image_augmenter_func = image_augmenter(
            config.color_aug_prob, cutout=config.use_cutout
        )
        self.lidar_augmenter_func = lidar_augmenter(
            config.lidar_aug_prob, cutout=config.use_cutout
        )

        self.forcast_step = int(config.forcast_time / (config.data_save_freq / config.carla_fps) + 0.5)

        # Initialize with 1 example per class
        self.angle_distribution = np.arange(len(config.angles)).tolist()
        self.speed_distribution = np.arange(len(config.target_speeds)).tolist()
        self.semantic_distribution = np.arange(len(config.semantic_weights)).tolist()
        total_routes = 0
        trainable_routes = 0
        skipped_routes = 0
        pruned_samples = 0

        # loops over the scenarios given in root (which is a list of the scenario folders)
        for sub_root in tqdm(root, file=sys.stdout, disable=rank != 0):

            # list subdirectories in root
            routes = next(os.walk(sub_root))[1]

            for (
                route
            ) in routes:  # loop over individual routes within this scenario folder
                total_routes += 1
                route_dir = sub_root + "/" + route
                lidar_dir = route_dir + "/" + SensorFolder.LIDAR.value

                # Skip repetitions we are not using
                repetition = int(re.search("_Rep(\\d+)", route).group(1))
                if repetition >= self.config.num_repetitions:
                    continue

                # Skip if we are doing validation on a non-validation town and vice versa
                town = int(re.search("Town(\\d+)", route).group(1))
                if self.validation and (town not in self.config.val_towns):
                    continue
                elif not self.validation and (town in self.config.val_towns):
                    continue

                # We skip data where the expert did not achieve perfect driving score (except for min speed infractions)
                if not self._is_valid_route(route_dir):
                    skipped_routes += 1
                    continue

                # Skip first frames so we can load them backward (for lidar and rgb len > skip_first)
                # temporal rgb/lidar goes backward in time, while seq_len goes forward
                first_frame = max(
                    config.img_seq_len * config.img_step_size,
                    config.lidar_seq_len * config.lidar_step_size,
                    config.skip_first,
                )

                num_seq = len(
                    os.listdir(lidar_dir)
                )  # How many frames recorded for the current route

                # Subtract the maximum of all the different forcasting times
                last_frame = num_seq - max(
                    0,
                    (self.config.seq_len - 1) * self.config.seq_step,
                    (self.config.trajectory_pred_len * self.config.trajectory_step_size) if self.config.use_trajectory_prediction else 0,
                    (self.config.pred_len * self.config.wp_dilation) if self.config.use_wp_gru else 0,
                )

                # Skip routes that are too short for a full seq
                if last_frame <= first_frame:
                    warnings.warn(f"Route was skipped due to not having enough frames for one full sequence (last_frame[{last_frame}]<=first_frame[{first_frame}]).")
                    skipped_routes += 1
                    continue

                trainable_routes += 1
                # For all frames of the route (that we want to load)
                for seq in range(first_frame, last_frame):
                    if seq % config.train_sampling_rate != 0:
                        continue

                    # Prune this sequence if it is deemed uninteresting
                    if self.heuristic_pruning:
                        measurements = []
                        start = seq if self.config.seq_len > 1 else seq-1
                        end = seq + self.config.seq_len * self.config.seq_step
                        for seq_i in range(start, end, self.config.seq_step):
                            measurement_file = route_dir + "/" + SensorFolder.MEASUREMENTS.value
                            measurements_i = self._load_json_gz(
                                measurement_file + f"/{seq_i :04}.json.gz"
                            )
                            measurements.append(measurements_i)

                        if not self._pruning_heuristic(measurements):
                            pruned_samples += 1
                            continue

                    # Store only the scenario + route + sample_start of the current index
                    # When loading, we only need to know where to start, and which route it is
                    scenario = sub_root.split("/")[-1]
                    self.route_root.append(f"{scenario}/{route}")
                    self.sample_start.append(seq)

                    if estimate_sem_distribution:
                        semantics_file = route_dir + "/" + SensorFolder.SEMANTICS.value + f"/{seq:04}.png"
                        semantics_i = self.converter[
                            self._load_png(semantics_file, crop=False)
                        ]  # pylint: disable=locally-disabled, unsubscriptable-object
                        self.semantic_distribution.extend(
                            semantics_i.flatten().tolist()
                        )

                    if estimate_class_distributions:
                        measurement = route_dir + "/" + SensorFolder.MEASUREMENTS.value

                        measurements_i = self._load_json_gz(
                            measurement + f"/{seq :04}.json.gz"
                        )

                        target_speed_index, angle_index = self.get_indices_speed_angle(
                            target_speed=measurements_i["target_speed"],
                            brake=measurements_i["brake"],
                            angle=measurements_i["angle"],
                        )

                        self.angle_distribution.append(angle_index)
                        self.speed_distribution.append(target_speed_index)

        if estimate_class_distributions:
            classes_target_speeds = np.unique(self.speed_distribution)
            target_speed_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes_target_speeds,
                y=self.speed_distribution,
            )

            config.target_speed_weights = target_speed_weights.tolist()
            print("config.target_speeds: ", config.target_speeds)
            print("config.target_speed_bins: ", config.target_speed_bins)
            print("classes_target_speeds: ", classes_target_speeds)
            print("Target speed weights: ", config.target_speed_weights)
            unique, counts = np.unique(self.speed_distribution, return_counts=True)
            ts_dict = dict(zip(unique, counts))
            print("Target speed counts: ", ts_dict)
            with open(f"ts_dict{len(unique)}.pickle", "wb") as handle:
                print("saving ts_dict")
                pickle.dump(ts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            classes_angles = np.unique(self.angle_distribution)
            angle_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes_angles,
                y=self.angle_distribution,
            )

            config.angle_weights = angle_weights.tolist()
            sys.exit()

        if estimate_sem_distribution:
            classes_semantic = np.unique(self.semantic_distribution)
            semantic_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes_semantic,
                y=self.semantic_distribution,
            )

            print("Semantic weights:", semantic_weights)

        del self.angle_distribution
        del self.speed_distribution
        del self.semantic_distribution

        # There is a complex "memory leak"/performance issue when using Python
        # objects like lists in a Dataloader that is loaded with
        # multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects
        # because they only have 1 refcount.
        self.route_root = np.array(self.route_root).astype(np.string_)
        self.sample_start = np.array(self.sample_start)
        if rank == 0 and len(root) != 0:
            print(f"Loading {len(self.route_root)} samples from {len(root)} scenarios")
            print("Total amount of routes:", total_routes)
            print("Skipped routes:", skipped_routes)
            print("Trainable routes:", trainable_routes)
            print(f"Maximum dataset size: {len(self.route_root) + pruned_samples}")
            print(f"Pruned {pruned_samples} samples ("
                f"{pruned_samples / max(len(self.route_root) + pruned_samples, 1):.1%} of dataset)")

    def __len__(self):
        """Returns the length of the dataset."""
        return self.sample_start.shape[0]

    def __getitem__(self, index):
        return self._get_data(index)

    def get_sensor_path(self, route_root: Union[str, os.PathLike], sensor: SensorFolder):
        """
        Get the path for a specific sensor folder.
        """
        return os.path.join(self.data_root, str(route_root, encoding="utf-8"), sensor.value)

    def _get_data(self, index):
        """
        Load the data at a specific index.
        """
        # Disable threading because the data loader will already split in processes.
        cv2.setNumThreads(0)

        data = {}

        route_root = self.route_root[index]
        sample_start = self.sample_start[index]

        images_root = self.get_sensor_path(route_root, SensorFolder.RGB)
        images_augmented_root = self.get_sensor_path(route_root, SensorFolder.RGB_AUGMENTED)
        semantics_root = self.get_sensor_path(route_root, SensorFolder.SEMANTICS)
        semantics_augmented_root = self.get_sensor_path(route_root, SensorFolder.SEMANTICS_AUGMENTED)
        bev_semantics_root = self.get_sensor_path(route_root, SensorFolder.BEV_SEMANTICS)
        bev_semantics_augmented_root = self.get_sensor_path(route_root, SensorFolder.BEV_SEMANTICS_AUGMENTED)
        depth_root = self.get_sensor_path(route_root, SensorFolder.DEPTH)
        depth_augmented_root = self.get_sensor_path(route_root, SensorFolder.DEPTH_AUGMENTED)
        lidars_root = self.get_sensor_path(route_root, SensorFolder.LIDAR)
        boxes_root = self.get_sensor_path(route_root, SensorFolder.BOXES)
        trajectories_root = self.get_sensor_path(route_root, SensorFolder.TRAJECTORIES)
        measurement_root = self.get_sensor_path(route_root, SensorFolder.MEASUREMENTS)

        if self.config.img_seq_len > 1:
            assert (
                self.config.seq_len == 1
            ), "img_seq_len>1 can only be used with seq_len=1"
        if self.config.lidar_seq_len > 1:
            assert (
                self.config.seq_len == 1
            ), "lidar_seq_len>1 can only be used with seq_len=1"

        # Since we load measurements for future time steps, we load and store them separately
        loaded_measurements = []
        start = sample_start
        end = start + self.config.seq_len * self.config.seq_step
        for seq_i in range(start, end, self.config.seq_step):
            measurement_file = measurement_root + f"/{seq_i:04}.json.gz"
            measurements_i = self._load_json_gz(measurement_file)
            loaded_measurements.append(measurements_i)

        # For lidar alignment, we need the current frame
        current_measurement = loaded_measurements[self.config.seq_len - 1]  # the present/current measurement

        # If using GRU WP prediction, append further future measurements
        if self.config.use_wp_gru:
            # Start: end of sequence
            # End: end of sequence + future time steps
            start = sample_start + (self.config.seq_len - 1) * self.config.seq_step
            end = start + self.config.pred_len * self.config.wp_dilation
            for seq_i in range(start, end, self.config.wp_dilation):
                measurement_file = measurement_root + f"/{seq_i:04}.json.gz"
                measurements_i = self._load_json_gz(measurement_file)
                loaded_measurements.append(measurements_i)

        # Below we load all inputs as sequences of frames
        # This would allow a model to process the data sequentially if seq_len > 1
        # To avoid complications, temporal rgb and temporal lidar are loaded in a separate loop
        # Temporal rgb and temporal lidar should not be used with seq_len > 1
        #   (it makes no sense to have a sequence of sequences of images or lidars...)

        # Model inputs
        image_seq = []
        lidar_seq = []
        speed_seq = []
        command_seq = []
        next_command_seq = []
        target_point_seq = []
        target_point_next_seq = []

        # Assume not aug, and update during temporal input loading
        apply_aug = False
        aug_rotation = 0
        aug_translation = 0
        images_path = images_root
        semantics_path = semantics_root
        bev_semantics_path = bev_semantics_root
        depth_path = depth_root

        # === Load model inputs ===
        # (rgb, lidar, ego_velocity, commmand and target_point)
        start = sample_start
        end = start + self.config.seq_len * self.config.seq_step
        for offset, seq_i in enumerate(range(start, end, self.config.seq_step)):
            measurement_i = loaded_measurements[offset]

            apply_aug = self.config.augment and random.random() <= self.config.augment_percentage
            # Augmentation is applied temporally
            # TODO: if we are loading temporal targets, we need to store a list of augment vs non-augment paths
            if apply_aug:
                aug_rotation = measurement_i["augmentation_rotation"]
                aug_translation = measurement_i["augmentation_translation"]
                images_path = images_augmented_root
                semantics_path = semantics_augmented_root
                bev_semantics_path = bev_semantics_augmented_root
                depth_path = depth_augmented_root

            # == Load image ==
            image_file = images_path + f"/{seq_i:04}.jpg"
            image_i = self._process_image(self._load_jpg(image_file))
            image_seq.append(image_i)

            # == Load lidar ==
            lidar_file = lidars_root + f"/{seq_i:04}.laz"
            lidar = self._load_lidar(lidar_file)
            # should not realign to the current measurement, since model doesn't know the future yet
            # That only makes sense if model gets the whole sequence at once (e.g. via video encoder)
            lidar = self.align(
                lidar,
                measurements_i,
                measurements_i,
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )
            lidar = self.lidar_to_histogram_features(
                lidar, use_ground_plane=self.config.use_ground_plane
            )
            lidar = self.lidar_augmenter_func(image=np.transpose(lidar, (1, 2, 0)))
            lidar = np.transpose(lidar, (2, 0, 1))
            lidar_seq.append(lidar)

            # == Load input measurements ==
            speed = measurement_i["speed"]
            speed_seq.append(speed)

            command = t_u.command_to_one_hot(measurement_i["command"])
            command_seq.append(command)
            next_command = t_u.command_to_one_hot(measurement_i["next_command"])
            next_command_seq.append(next_command)

            target_point = self.augment_target_point(
                np.array(measurement_i["target_point"]),
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )
            target_point_seq.append(target_point)

            target_point_next = self.augment_target_point(
                np.array(measurement_i["target_point_next"]),
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )
            target_point_next_seq.append(target_point_next)

        # Load as np.arrays, preserves the seq dimension
        data["rgb"] = np.array(image_seq)
        data["lidar"] = np.array(lidar_seq)
        data["speed"] = np.array(speed_seq)
        data["command"] = np.array(command_seq)
        data["next_command"] = np.array(next_command_seq)
        data["target_point"] = np.array(target_point_seq)
        data["target_point_next"] = np.array(target_point_next_seq)

        # If not using sequences, remove the seq dimension
        if self.config.seq_len == 1:
            data = {k: v.squeeze(0) for k, v in data.items()}

        # Model targets
        semantics_seq = []
        bev_semantics_seq = []
        depth_seq = []
        boxes_seq = []
        boxes_raw_seq = []  # needed for trajectories
        future_boxes_seq = []
        box_targets_seq = []
        avg_factor_seq = []
        waypoints_seq = []
        route_seq = []
        brake_seq = []
        angle_index_seq = []
        target_speed_seq = []
        target_speed_twohot_seq = []

        # Other targets (for compatibility)
        steer_seq = []
        throttle_seq = []
        light_hazard_seq = []
        stop_sign_hazard_seq = []
        junction_seq = []
        speed_seq = []
        theta_seq = []

        # === Load model targets ===
        # (semantic, depth, bev_semantic, depth, box targets, route, trajectories, measurement data)
        # TODO: could be useful to load temporal targets also
        #       currently only loads the last seq_i 
        #       ^ also see todo above ^
        #       (need to apply augments correctly, then)
        #       also needs to get correct measurement_i
        start = sample_start + (self.config.seq_len - 1) * self.config.seq_step
        end = start + 1
        for offset, seq_i in enumerate(range(start, end, self.config.seq_step)):
            measurement_i = loaded_measurements[-1]

            # == Load semantics ==
            if self.config.use_semantic:
                semantics_file = semantics_path + f"/{seq_i:04}.png"
                semantic_i = self._process_semantics(self._load_png(semantics_file))
                semantics_seq.append(semantic_i)

            # == Load BEV semantics == 
            if self.config.use_bev_semantic:
                bev_semantics_file = bev_semantics_path + f"/{seq_i:04}.png"
                bev_semantic_i = self._process_bev_semantics(
                    self._load_png(bev_semantics_file, crop=False)
                )
                bev_semantics_seq.append(bev_semantic_i)

            # == Load depth ==
            if self.config.use_depth:
                depth_file = depth_path + f"/{seq_i:04}.png"
                depth_i = self._process_depth(self._load_png(depth_file))
                depth_seq.append(depth_i)

            # == Load boxes ==
            if self.config.detect_boxes:
                boxes_i = future_boxes_i = None

                box_file = boxes_root + f"/{seq_i:04}.json.gz"
                boxes_i = self._load_json_gz(box_file)

                boxes_raw_seq.append(boxes_i)

                if self.config.use_plant:
                    future_box_file = boxes_root + f"/{seq_i + self.forcast_step:04}.json.gz"
                    future_boxes_i = self._load_json_gz(future_box_file)

                # Process and pad the boxes
                boxes_i, boxes_padded_i, _, future_boxes_padded_i = self._process_boxes(
                    boxes_i, future_boxes_i, aug_translation, aug_rotation
                )
                boxes_seq.append(boxes_padded_i)
                if future_boxes_i is not None:
                    future_boxes_seq.append(future_boxes_padded_i)

                # Get the targets from the current boxes
                target_result, avg_factor = self.get_targets(
                    boxes_i,
                    self.config.lidar_resolution_height
                    // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width
                    // self.config.bev_down_sample_factor,
                )
                box_targets_seq.append(target_result)
                avg_factor_seq.append(avg_factor)

            # == Load waypoints
            if self.config.use_wp_gru:
                waypoints = self.get_waypoints(
                    loaded_measurements[offset:offset+self.config.pred_len],
                    y_augmentation=aug_translation,
                    yaw_augmentation=aug_rotation
                )
                waypoints_seq.append(waypoints)

            # == Load route ==
            route = measurement_i["route"]
            if len(route) < self.config.num_route_points:
                num_missing = self.config.num_route_points - len(route)
                route = np.array(route)
                # Fill the empty spots by repeating the last point.
                route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
            else:
                route = np.array(route[: self.config.num_route_points])

            route = self.augment_route(
                route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation
            )
            if self.config.smooth_route:
                route = self.smooth_path(route)
            route_seq.append(route)

            # == Load brake, target speed and angle index ==
            brake = measurement_i["brake"]
            brake_seq.append(brake)

            target_speed_index, angle_index = self.get_indices_speed_angle(
                target_speed=measurement_i["target_speed"],
                brake=brake,
                angle=measurement_i["angle"],
            )
            target_speed_seq.append(target_speed_index)
            angle_index_seq.append(angle_index)

            target_speed_twohot = self.get_two_hot_encoding(
                measurement_i["target_speed"], self.config.target_speeds, brake
            )
            target_speed_twohot_seq.append(target_speed_twohot)

            # == Load other measurements ==
            steer_seq = measurement_i["steer"]
            throttle_seq = measurement_i["throttle"]
            light_hazard_seq = measurement_i["light_hazard"]
            stop_sign_hazard_seq = measurement_i["stop_sign_hazard"]
            junction_seq = measurement_i["junction"]
            speed_seq = measurement_i["speed"]
            theta_seq = measurement_i["theta"]


        # Gather into data
        target_data = {}
        if self.config.use_semantic:
            target_data["semantic"] = np.array(semantics_seq)
        if self.config.use_bev_semantic:
            target_data["bev_semantic"] = np.array(bev_semantics_seq)
        if self.config.use_depth:
            target_data["depth"] = np.array(depth_seq)
        if self.config.detect_boxes:
            target_data["bounding_boxes"] = np.array(boxes_seq)
            if self.config.use_plant:
                target_data["future_bounding_boxes"] = np.array(future_boxes_seq)
            target_data["center_heatmap"] = np.array(
                [t["center_heatmap_target"] for t in box_targets_seq]
            )
            target_data["wh"] = np.array([t["wh_target"] for t in box_targets_seq])
            target_data["yaw_class"] = np.array([t["yaw_class_target"] for t in box_targets_seq])
            target_data["yaw_res"] = np.array([t["yaw_res_target"] for t in box_targets_seq])
            target_data["offset"] = np.array([t["offset_target"] for t in box_targets_seq])
            target_data["velocity"] = np.array([t["velocity_target"] for t in box_targets_seq])
            target_data["brake_target"] = np.array([t["brake_target"] for t in box_targets_seq])
            target_data["pixel_weight"] = np.array([t["pixel_weight"] for t in box_targets_seq])
            target_data["avg_factor"] = np.array(avg_factor_seq)
        if self.config.use_wp_gru:
            target_data['ego_waypoints'] = np.array(waypoints_seq)
        target_data["route"] = np.array(route_seq)
        target_data["brake"] = np.array(brake_seq)
        target_data["angle_index"] = np.array(angle_index_seq)
        target_data["target_speed"] = np.array(target_speed_seq)
        target_data["target_speed_twohot"] = np.array(target_speed_twohot_seq)

        target_data["steer"] = np.array(steer_seq)
        target_data["throttle"] = np.array(throttle_seq)
        target_data["light"] = np.array(light_hazard_seq)
        target_data["stop_sign"] = np.array(stop_sign_hazard_seq)
        target_data["junction"] = np.array(junction_seq)
        target_data["theta"] = np.array(theta_seq)

        # If not using sequences for targets, remove the seq dimension
        if target_data["brake"].shape[0] == 1:  # e.g., brake
            target_data = {k: v.squeeze(0) for k, v in target_data.items()}

        # Add targets to data
        data.update(target_data)

        # == Load temporal images ==
        if self.config.img_seq_len > 1:
            temporal_images = []
            # Temporal data just for images, loaded from the past to the present
            # Load all images with the same augmentation, to avoid complications
            start = sample_start - self.config.img_seq_len * self.config.img_step_size
            end = start + self.config.img_seq_len * self.config.img_step_size
            for seq_i in range(start, end, self.config.img_step_size):
                img_file = images_path + f"/{seq_i:04}.jpg"  # images path means augmented or not (random)
                temporal_image_i = self._process_image(self._load_jpg(img_file))
                temporal_images.append(temporal_image_i)

            data["temporal_rgb"] = np.array(temporal_images)

        # == Load temporal lidar ==
        if self.config.lidar_seq_len > 1:
            temporal_lidars = []
            # Temporal data just for LiDAR, loaded from the past to the present
            start = sample_start - self.config.lidar_seq_len * self.config.lidar_step_size
            end = start + self.config.lidar_seq_len * self.config.lidar_step_size
            for seq_i in range(start, end, self.config.lidar_step_size):
                measurement_file = measurement_root + f"/{seq_i:04}.json.gz"
                lidar_file = lidars_root + f"/{seq_i:04}.laz"

                temporal_measurements_i = self._load_json_gz(measurement_file)
                temporal_lidar_i = self._load_lidar(lidar_file)

                reference = current_measurement if self.config.realign_lidar else temporal_measurements_i
                temporal_lidar_i = self.align(
                    temporal_lidar_i,
                    temporal_measurements_i,
                    reference,
                    y_augmentation=aug_translation,
                    yaw_augmentation=aug_rotation,
                )

                temporal_lidar_i = self.lidar_to_histogram_features(
                    temporal_lidar_i, use_ground_plane=self.config.use_ground_plane
                )
                temporal_lidars.append(temporal_lidar_i)

            temporal_lidar_bev = np.concatenate(temporal_lidars, axis=0)
            temporal_lidar_bev = self.lidar_augmenter_func(
                image=np.transpose(temporal_lidar_bev, (1, 2, 0))
            )
            data["temporal_lidar"] = np.transpose(temporal_lidar_bev, (2, 0, 1))

        # == Load trajectories ==
        # NOTE: this only loads boxes for the last timestep, and predicts the future
        # If using temporal targets, this needs to be changed
        if self.config.use_trajectory_prediction:
            trajectories = []
            # Load future samples data (from current+1)
            start = sample_start + 1 * self.config.trajectory_step_size
            end = start + self.config.trajectory_pred_len * self.config.trajectory_step_size
            for seq_i in range(start, end, self.config.trajectory_step_size):
                # We use future bounding boxes to parse the trajectories
                trajectory_file = trajectories_root + f"/{seq_i:04}.json.gz"
                future_boxes_i = self._load_json_gz(trajectory_file)
                trajectories.append(future_boxes_i)

            if self.config.detect_boxes:
                current_boxes = boxes_raw_seq[-1]
            else:
                # Load *current* box
                box_path = boxes_root + f"/{start-1:04}.json.gz"
                current_boxes = self._load_json_gz(box_path)

            traj, mask = self._process_trajectories(
                trajectories,
                current_boxes=current_boxes,
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )
            data["trajectories"] = traj
            data["trajectories_mask"] = mask

        return data

    def _process_image(self, image_array: np.ndarray) -> np.ndarray:
        if self.config.use_color_aug:
            image_array = self.image_augmenter_func(image=image_array)
        image_array = np.transpose(image_array, (2, 0, 1))
        return image_array

    def _process_semantics(self, semantics_array: np.ndarray) -> np.ndarray:
        semantics_array = self.converter[semantics_array]
        semantics_array = semantics_array[
            :: self.config.perspective_downsample_factor,
            :: self.config.perspective_downsample_factor,
        ]
        return semantics_array

    def _process_bev_semantics(self, bev_array: np.ndarray) -> np.ndarray:
        # NOTE the BEV label can unfortunately only be saved up to 2.0 ppm resolution. We upscale it here.
        # If you change these values you might need to change the up-scaling as well.
        assert self.config.pixels_per_meter == 4.0
        assert self.config.pixels_per_meter_collection == 2.0
        assert self.config.lidar_resolution_width == 256
        assert self.config.lidar_resolution_height == 256
        assert self.config.max_x == 32
        assert self.config.min_x == -32
        if self.config.pixels_per_meter == 4.0:
            # Downsample BEV
            bev_array = bev_array[64:192, 64:192].repeat(2, axis=0).repeat(2, axis=1)
        bev_array = self.bev_converter[bev_array]
        return bev_array

    def _process_depth(self, depth_array: np.ndarray) -> np.ndarray:
        depth_array = depth_array.astype(np.float32) / 255.0
        depth_array = cv2.resize(
            depth_array,
            dsize=(
                depth_array.shape[1] // self.config.perspective_downsample_factor,
                depth_array.shape[0] // self.config.perspective_downsample_factor,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        return depth_array

    def _process_trajectories(self, temporal_boxes, current_boxes, y_augmentation=0.0, yaw_augmentation=0.0):
        # TODO: make sure the current boxes are augmented correctly

        # N: max number of boxes
        # n: number of actual boxes
        # F: future timesteps
        N = self.config.max_num_trajectories
        F = self.config.trajectory_pred_len

        # Only used to get the bounding box IDs
        # Remove boxes outside bounds after aug
        # These *should* now correspond to the BB targets from self._process_boxes(...)
        _, _, valid_boxes = self.parse_bounding_boxes(
            current_boxes, None,
            y_augmentation=y_augmentation, yaw_augmentation=yaw_augmentation, return_original=True
        )
        valid_ids = list(set([b['id'] for b in valid_boxes]))

        # Loop over all timesteps
        all_timesteps = []  # id->pos mapping across all timesteps F
        for t, boxes in enumerate(temporal_boxes):  # [F, n, :]

            # Parse the boxes
            parsed_boxes, orig_boxes = self.parse_bounding_boxes_traj(
                future_boxes=boxes,  # current (0th) timestep future
                reference=current_boxes,
                y_augmentation=y_augmentation,
                yaw_augmentation=yaw_augmentation
            )

            # For all actors in this timestep, get their id and their relative position
            id_to_xy = {}
            for box_data, box in zip(parsed_boxes, orig_boxes):
                x, y = box_data[:2]
                if box['id'] in valid_ids:
                    if t != 0 or not y > 255:
                        id_to_xy[box['id']] = x, y
            all_timesteps.append(id_to_xy)

        # Create a mask for valid trajectory points,
        # some actors disappear/reappear in future timesteps (bc. random bb selection), mask those points out
        mask = np.zeros((N, F))

        # For all actors (up to a limit), for all timesteps, get the position of that actor at that timestep
        # If position is found, append position and set mask to 1
        # Else, append zero and set mask to 0
        n = valid_ids.__len__()
        xy_lim = np.array([  # mark oob as zero mask
            [self.config.trajectory_min_x, self.config.trajectory_min_y], 
            [self.config.trajectory_max_x, self.config.trajectory_max_y]
        ])
        all_trajectories = np.zeros((max(n, N), F, 2))  # [n|N, F, 2]
        for actor_n, actor in enumerate(valid_ids): # [min(n,N)], truncate if >N
            actor_traj = np.zeros((F, 2))
            for t, id_to_pos in enumerate(all_timesteps):  # [F, n]
                pos = id_to_pos.get(actor, None)
                # If actor does exist in this timestep, set point and mask
                if pos is not None:
                    actor_traj[t] = pos
                    # Only valid if within bounds
                    if (pos > xy_lim[0]).all() and (pos < xy_lim[1]).all():
                        mask[actor_n, t] = 1

            all_trajectories[actor_n] = actor_traj

        all_trajectories = all_trajectories[: N]  # truncate to max num trajs (N)

        # Finally, normalize the trajectories
        all_trajectories = self.normalize_trajectories(all_trajectories)

        # These are now normalized, Ego_0-aligned trajectories for all actors visible in 0th timestep
        return all_trajectories, mask

    def normalize_trajectories(self, trajectory):
        max_xy = np.array([self.config.trajectory_max_x, self.config.trajectory_max_y])
        min_xy = np.array([self.config.trajectory_min_x, self.config.trajectory_min_y])
        trajectory = trajectory - min_xy
        trajectory = trajectory / (max_xy - min_xy)
        return trajectory

    def _process_boxes(self, boxes_i, future_boxes_i, aug_translation, aug_rotation):
        bounding_boxes, future_bounding_boxes = self.parse_bounding_boxes(
            boxes_i,
            future_boxes_i,
            y_augmentation=aug_translation,
            yaw_augmentation=aug_rotation,
        )

        # Pad bounding boxes to a fixed number
        bounding_boxes = np.array(bounding_boxes)
        bounding_boxes_padded = np.zeros((self.config.max_num_bbs, 8), dtype=np.float32)
        future_bounding_boxes_padded = None

        if self.config.use_plant:
            future_bounding_boxes = np.array(future_bounding_boxes)
            future_bounding_boxes_padded = (
                np.ones((self.config.max_num_bbs, 8), dtype=np.int32)
                * self.config.ignore_index
            )

        if bounding_boxes.shape[0] > 0:
            if bounding_boxes.shape[0] <= self.config.max_num_bbs:
                bounding_boxes_padded[: bounding_boxes.shape[0], :] = bounding_boxes
                if self.config.use_plant:
                    future_bounding_boxes_padded[
                        : future_bounding_boxes.shape[0], :
                    ] = future_bounding_boxes
            else:
                bounding_boxes_padded[: self.config.max_num_bbs, :] = bounding_boxes[
                    : self.config.max_num_bbs
                ]
                if self.config.use_plant:
                    future_bounding_boxes_padded[: self.config.max_num_bbs, :] = (
                        future_bounding_boxes[: self.config.max_num_bbs]
                    )
        return (
            bounding_boxes,
            bounding_boxes_padded,
            future_bounding_boxes,
            future_bounding_boxes_padded,
        )

    def _is_valid_route(self, route_dir: Union[os.PathLike, str]) -> bool:
        """
        Returns True if the route meets the success conditions (perfect score, not failed, etc.)
        """
        if route_dir.split("/")[-1].startswith("FAILED_") or not os.path.isfile(
            route_dir + "/results.json.gz"
        ):
            return False
        lidar_dir = route_dir + "/lidar"
        if not os.path.exists(lidar_dir):
            return False
        with gzip.open(route_dir + "/results.json.gz", "rt", encoding="utf-8") as f:
            results_route = ujson.load(f)
        condition1 = results_route["scores"]["score_composed"] < 100.0 and not (
            results_route["num_infractions"]
            == len(results_route["infractions"]["min_speed_infractions"])
        )
        condition2 = results_route["status"] == "Failed - Agent couldn't be set up"
        condition3 = results_route["status"] == "Failed"
        condition4 = results_route["status"] == "Failed - Simulation crashed"
        condition5 = results_route["status"] == "Failed - Agent crashed"
        return not (condition1 or condition2 or condition3 or condition4 or condition5)

    def _pruning_heuristic(self, measurements: list[dict]):
        speed_threshold = 0.1
        angle_threshold = 0.5 * (np.pi / 180)

        speeds = np.array([m['target_speed'] for m in measurements])
        route_points = np.array([
            [p for p in m['route'][:self.config.predict_checkpoint_len]]
            for m in measurements
        ])  # Shape: (seq_len, checkpoints, 2)

        # Check angle changes
        y = route_points[:, :, 1]
        x = route_points[:, :, 0]
        route_angles = np.arctan2(y, x)  # angle of point (relative to ego orientation) (seq_len, checkpoints)
        angle_diffs = np.abs(np.diff(route_angles, axis=0))  # (seq_len-1, checkpoints)
        condition1 = np.any(angle_diffs > angle_threshold)

        # Check speed changes
        speed_diffs = np.abs(np.diff(speeds))
        condition2 = np.any(speed_diffs > speed_threshold)

        return condition1 or condition2 or (random.random() <= 0.14)

    def get_targets(self, gt_bboxes, feat_h, feat_w):
        """
        Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with shape (num_gts, 4)
              in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset predict, shape (B, 2, H, W).
        """
        img_h = self.config.lidar_resolution_height
        img_w = self.config.lidar_resolution_width

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = np.zeros(
            [self.config.num_bb_classes, feat_h, feat_w], dtype=np.float32
        )
        wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
        offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
        yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
        yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
        brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
        pixel_weight = np.zeros(
            [2, feat_h, feat_w], dtype=np.float32
        )  # 2 is the max of the channels above here.

        if not gt_bboxes.shape[0] > 0:
            target_result = {
                "center_heatmap_target": center_heatmap_target,
                "wh_target": wh_target,
                "yaw_class_target": yaw_class_target.squeeze(0),
                "yaw_res_target": yaw_res_target,
                "offset_target": offset_target,
                "velocity_target": velocity_target,
                "brake_target": brake_target.squeeze(0),
                "pixel_weight": pixel_weight,
            }
            return target_result, 1

        center_x = gt_bboxes[:, [0]] * width_ratio
        center_y = gt_bboxes[:, [1]] * height_ratio
        gt_centers = np.concatenate((center_x, center_y), axis=1)

        for j, ct in enumerate(gt_centers):
            ctx_int, cty_int = ct.astype(int)
            ctx, cty = ct
            extent_x = gt_bboxes[j, 2] * width_ratio
            extent_y = gt_bboxes[j, 3] * height_ratio

            radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
            radius = max(2, int(radius))
            ind = gt_bboxes[j, -1].astype(int)

            # print(center_heatmap_target[ind], [ctx_int, cty_int], radius)
            g_t.gen_gaussian_target(
                center_heatmap_target[ind], [ctx_int, cty_int], radius
            )

            wh_target[0, cty_int, ctx_int] = extent_x
            wh_target[1, cty_int, ctx_int] = extent_y

            yaw_class, yaw_res = angle2class(gt_bboxes[j, 4], self.config.num_dir_bins)

            yaw_class_target[0, cty_int, ctx_int] = yaw_class
            yaw_res_target[0, cty_int, ctx_int] = yaw_res

            velocity_target[0, cty_int, ctx_int] = gt_bboxes[j, 5]
            # Brakes can potentially be continous but we classify them now.
            # Using mathematical rounding the split is applied at 0.5
            brake_target[0, cty_int, ctx_int] = int(round(gt_bboxes[j, 6]))

            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int
            # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
            # Used to ignore the pixels without bbs in the loss.
            pixel_weight[:, cty_int, ctx_int] = 1.0

        avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
        target_result = {
            "center_heatmap_target": center_heatmap_target,
            "wh_target": wh_target,
            "yaw_class_target": yaw_class_target.squeeze(0),
            "yaw_res_target": yaw_res_target,
            "offset_target": offset_target,
            "velocity_target": velocity_target,
            "brake_target": brake_target.squeeze(0),
            "pixel_weight": pixel_weight,
        }
        return target_result, avg_factor

    def augment_route(self, route, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array(
            [
                [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
            ]
        )

        translation = np.array([[0.0, y_augmentation]])
        route_aug = (rotation_matrix.T @ (route - translation).T).T
        return route_aug

    def augment_target_point(
        self, target_point, y_augmentation=0.0, yaw_augmentation=0.0
    ):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array(
            [
                [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
            ]
        )

        translation = np.array([[0.0], [y_augmentation]])
        pos = np.expand_dims(target_point, axis=1)
        target_point_aug = rotation_matrix.T @ (pos - translation)
        return np.squeeze(target_point_aug)

    def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
        """transform waypoints to be origin at ego_matrix"""
        origin = measurements[0]
        origin_matrix = np.array(origin["ego_matrix"])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        waypoints = []
        for index in range(self.config.seq_len, len(measurements)):
            waypoint = np.array(measurements[index]["ego_matrix"])[:3, 3:4]
            waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
            # Drop the height dimension because we predict waypoints in BEV
            waypoints.append(waypoint_ego_frame[:2, 0])

        # Data augmentation
        waypoints_aug = []
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array(
            [
                [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
            ]
        )

        translation = np.array([[0.0], [y_augmentation]])
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))

        return waypoints_aug

    def align(
        self,
        lidar_0,
        measurements_0,
        measurements_1,
        y_augmentation=0.0,
        yaw_augmentation=0,
    ):
        """
        Converts the LiDAR from the coordinate system of measurements_0 to the
        coordinate system of measurements_1. In case of data augmentation, the
        shift of y and rotation around the yaw are taken into account, such that the
        LiDAR is in the same coordinate system as the rotated camera.
        :param lidar_0: (N,3) numpy, LiDAR point cloud
        :param measurements_0: measurements describing the coordinate system of the LiDAR
        :param measurements_1: measurements describing the target coordinate system
        :param y_augmentation: Data augmentation shift in meters
        :param yaw_augmentation: Data augmentation rotation in degree
        :return: (N,3) numpy, Converted LiDAR
        """
        pos_1 = np.array(
            [measurements_1["pos_global"][0], measurements_1["pos_global"][1], 0.0]
        )
        pos_0 = np.array(
            [measurements_0["pos_global"][0], measurements_0["pos_global"][1], 0.0]
        )
        pos_diff = pos_1 - pos_0
        rot_diff = t_u.normalize_angle(
            measurements_1["theta"] - measurements_0["theta"]
        )

        # Rotate difference vector from global to local coordinate system.
        rotation_matrix = np.array(
            [
                [
                    np.cos(measurements_1["theta"]),
                    -np.sin(measurements_1["theta"]),
                    0.0,
                ],
                [np.sin(measurements_1["theta"]), np.cos(measurements_1["theta"]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        pos_diff = rotation_matrix.T @ pos_diff

        lidar_1 = t_u.algin_lidar(lidar_0, pos_diff, rot_diff)

        pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
        rot_diff_aug = np.deg2rad(yaw_augmentation)

        lidar_1_aug = t_u.algin_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

        return lidar_1_aug

    def lidar_to_histogram_features(self, lidar, use_ground_plane):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
        :param lidar: (N,3) numpy, LiDAR point cloud
        :param use_ground_plane, whether to use the ground plane
        :return: (2, H, W) numpy, LiDAR as sparse image
        """

        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self.config.min_x,
                self.config.max_x,
                (self.config.max_x - self.config.min_x)
                * int(self.config.pixels_per_meter)
                + 1,
            )
            ybins = np.linspace(
                self.config.min_y,
                self.config.max_y,
                (self.config.max_y - self.config.min_y)
                * int(self.config.pixels_per_meter)
                + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self.config.hist_max_per_pixel] = self.config.hist_max_per_pixel
            overhead_splat = hist / self.config.hist_max_per_pixel
            # The transpose here is an efficient axis swap.
            # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
            # (x height channel, y width channel)
            return overhead_splat.T

        # Remove points above the vehicle
        lidar = lidar[lidar[..., 2] < self.config.max_height_lidar]
        below = lidar[lidar[..., 2] <= self.config.lidar_split_height]
        above = lidar[lidar[..., 2] > self.config.lidar_split_height]
        below_features = splat_points(below)
        above_features = splat_points(above)
        if use_ground_plane:
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features

    def get_bbox_label(self, bbox_dict, y_augmentation=0.0, yaw_augmentation=0):
        # augmentation
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array(
            [
                [np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)],
            ]
        )

        position = np.array([[bbox_dict["position"][0]], [bbox_dict["position"][1]]])
        translation = np.array([[0.0], [y_augmentation]])

        position_aug = rotation_matrix.T @ (position - translation)

        x, y = position_aug[:2, 0]
        # center_x, center_y, w, h, yaw
        bbox = np.array(
            [x, y, bbox_dict["extent"][0], bbox_dict["extent"][1], 0, 0, 0, 0]
        )
        bbox[4] = t_u.normalize_angle(bbox_dict["yaw"] - aug_yaw_rad)

        if bbox_dict["class"] == "car":
            bbox[5] = bbox_dict["speed"]
            # check for nans
            if np.isnan(bbox_dict["brake"]):
                bbox[6] = 0
            else:
                bbox[6] = bbox_dict["brake"]
            if bbox_dict["role_name"] == "scenario" and bbox_dict["type_id"] in [
                "vehicle.dodge.charger_police_2020",
                "vehicle.dodge.charger_police",
                "vehicle.ford.ambulance",
                "vehicle.carlamotors.firetruck",
            ]:
                # this is an emergency vehicle that we need to yield to (or dodge in the RunningRedLight scenario)
                bbox[7] = 4
            else:
                bbox[7] = 0
        elif bbox_dict["class"] == "walker":
            bbox[5] = bbox_dict["speed"]
            bbox[7] = 1
        elif bbox_dict["class"] == "traffic_light":
            bbox[7] = 2
        elif bbox_dict["class"] == "stop_sign":
            bbox[7] = 3
        return bbox, bbox_dict["position"][2]

    def parse_bounding_boxes(
        self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0, return_original=False
    ):

        if self.config.use_plant and future_boxes is not None:
            # Find ego matrix of the current time step, i.e. the coordinate frame we want to use:
            ego_matrix = None
            ego_yaw = None
            # ego_car always exists
            for ego_candiate in boxes:
                if ego_candiate["class"] == "ego_car":
                    ego_matrix = np.array(ego_candiate["matrix"])
                    ego_yaw = t_u.extract_yaw_from_matrix(ego_matrix)
                    break
        original_boxes = []
        bboxes = []
        future_bboxes = []
        for idx, current_box in enumerate(boxes):
            if current_box["class"] not in [
                "traffic_light",
                "stop_sign",
                "car",
                "walker",
            ]:
                continue  # Ignore any additional bbs such as ego and static. Might make sense to add static, haven't tested it.

            bbox, height = self.get_bbox_label(
                current_box, y_augmentation, yaw_augmentation
            )

            if "num_points" in current_box:
                if (
                    current_box["class"] == "walker"
                    and current_box["num_points"]
                    <= self.config.num_lidar_hits_for_detection_walker
                    or current_box["class"] == "car"
                    and current_box["num_points"]
                    <= self.config.num_lidar_hits_for_detection_car
                ):
                    continue
            if current_box["class"] == "traffic_light":
                # Only use/detect boxes that are red and affect the ego vehicle
                if not current_box["affects_ego"] or current_box["state"] == "Green":
                    continue

            if current_box["class"] == "stop_sign":
                # Don't detect cleared stop signs.
                if not current_box["affects_ego"]:
                    continue

            # Filter bb that are outside of the LiDAR after the augmentation.
            if (
                bbox[0] <= self.config.min_x
                or bbox[0] >= self.config.max_x
                or bbox[1] <= self.config.min_y
                or bbox[1] >= self.config.max_y
                or height <= self.config.min_z
                or height >= self.config.max_z
            ):
                continue

            # Load bounding boxes to forcast
            if self.config.use_plant and future_boxes is not None:
                exists = False
                for future_box in future_boxes:
                    # We only forecast boxes visible in the current frame
                    if future_box["id"] == current_box["id"] and future_box[
                        "class"
                    ] in ("car", "walker"):
                        # Found a valid box
                        # Get values in current coordinate system
                        future_box_matrix = np.array(future_box["matrix"])
                        relative_pos = t_u.get_relative_transform(
                            ego_matrix, future_box_matrix
                        )
                        # Update position into current coordinate system
                        future_box["position"] = [
                            relative_pos[0],
                            relative_pos[1],
                            relative_pos[2],
                        ]
                        future_yaw = t_u.extract_yaw_from_matrix(future_box_matrix)
                        relative_yaw = t_u.normalize_angle(future_yaw - ego_yaw)
                        future_box["yaw"] = relative_yaw

                        converted_future_box, _ = self.get_bbox_label(
                            future_box, y_augmentation, yaw_augmentation
                        )
                        quantized_future_box = self.quantize_box(converted_future_box)
                        future_bboxes.append(quantized_future_box)
                        exists = True
                        break

                if not exists:
                    # Bounding box has no future counterpart. Add a dummy with ignore index
                    future_bboxes.append(
                        np.array(
                            [
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                                self.config.ignore_index,
                            ]
                        )
                    )

            if not self.config.use_plant:
                bbox = t_u.bb_vehicle_to_image_system(
                    bbox,
                    self.config.pixels_per_meter,
                    self.config.min_x,
                    self.config.min_y,
                )
            if return_original:
                original_boxes.append(current_box)

            bboxes.append(bbox)

        if return_original:
            return bboxes, future_bboxes, original_boxes
        return bboxes, future_bboxes

    def parse_bounding_boxes_traj(self, future_boxes, reference, y_augmentation=0.0, yaw_augmentation=0):

        # We need to find the ego matrix and yaw
        ego_matrix = None
        ego_yaw = None
        # ego_car always exists
        for ego_candiate in reference:
            if ego_candiate["class"] == "ego_car":
                ego_matrix = np.array(ego_candiate["matrix"])
                ego_yaw = t_u.extract_yaw_from_matrix(ego_matrix)
                break

        bboxes = []
        original_boxes = []
        for idx, sample_box in enumerate(future_boxes):
            # (1) Filter out non-valid boxes        
            # Only detect movable objects
            condition1 = sample_box["class"] in ["car", "walker"]
            # Only detect boxes with enough lidar hits
            condition2 = ("num_points" not in sample_box
                    or not (
                        # Walker under walker threshold
                        (sample_box["class"] == "walker"
                            and sample_box.get("num_points", -1)
                            <= self.config.num_lidar_hits_for_detection_walker)
                        # Or car under car threshold
                        or (sample_box["class"] == "car"
                            and sample_box.get("num_points", -1)
                            <= self.config.num_lidar_hits_for_detection_car)
                    )
            )
            if not (condition1 and condition2):
                continue  # Ignore box

            # (2) Find the relative position (P_{E_t} -> P_{E_0})
            # (P_{E_t} is relative to ego at t, P_{E_0} is relative to ego at 0)
            sample_matrix = np.array(sample_box["matrix"])
            relative_pos = t_u.get_relative_transform(
                ego_matrix, sample_matrix
            )
            sample_yaw = t_u.extract_yaw_from_matrix(sample_matrix)
            relative_yaw = t_u.normalize_angle(sample_yaw - ego_yaw)
            sample_box["position"] = [*relative_pos[:3]]
            sample_box["yaw"] = relative_yaw

            # (3) Convert to label array
            # All future positions should be relative to t=0 aug
            bbox, height = self.get_bbox_label(
                sample_box, y_augmentation, yaw_augmentation
            )

            # (4) Transform to image system
            bbox = t_u.bb_vehicle_to_image_system(
                    bbox,
                    self.config.pixels_per_meter,
                    self.config.min_x,
                    self.config.min_y,
            )

            original_boxes.append(sample_box)
            bboxes.append(bbox)

        return bboxes, original_boxes

    def quantize_box(self, boxes):
        """Quantizes a bounding box into bins and writes the index into the array a classification label"""
        # range of xy is [-32, 32]
        # range of yaw is [-pi, pi]
        # range of speed is [0, 60]
        # range of extent is [0, 30]

        # Normalize all values between 0 and 1
        boxes[0] = (boxes[0] + self.config.max_x) / (
            self.config.max_x - self.config.min_x
        )
        boxes[1] = (boxes[1] + self.config.max_y) / (
            self.config.max_y - self.config.min_y
        )

        # quantize extent
        boxes[2] = boxes[2] / 30
        boxes[3] = boxes[3] / 30

        # quantize yaw
        boxes[4] = (boxes[4] + np.pi) / (2 * np.pi)

        # quantize speed, convert max speed to m/s
        boxes[5] = boxes[5] / (self.config.plant_max_speed_pred / 3.6)

        # 6 Brake is already in 0, 1
        # Clip values that are outside the range we classify
        boxes[:7] = np.clip(boxes[:7], 0, 1)

        size_pos = pow(2, self.config.plant_precision_pos)
        size_speed = pow(2, self.config.plant_precision_speed)
        size_angle = pow(2, self.config.plant_precision_angle)

        boxes[[0, 1, 2, 3]] = (boxes[[0, 1, 2, 3]] * (size_pos - 1)).round()
        boxes[4] = (boxes[4] * (size_angle - 1)).round()
        boxes[5] = (boxes[5] * (size_speed - 1)).round()
        boxes[6] = boxes[6].round()

        return boxes.astype(np.int32)

    def get_two_hot_encoding(self, target_speed, config_target_speeds, brake):
        if target_speed < 0:
            raise ValueError(
                "Target speed value must be non-negative for two-hot encoding."
            )
        # Calculate two-hot labes as described in https://arxiv.org/pdf/2403.03950.pdf
        label = np.zeros((len(config_target_speeds),))
        if brake:
            label[0] = 1.0
        else:
            if not np.any(
                np.array(config_target_speeds) > target_speed
            ):  # value is in the last bin (which goes to infinity)
                label[-1] = 1.0
            else:
                upper_ind = np.argmax(np.array(config_target_speeds) > target_speed)
                lower_ind = upper_ind - 1
                lower_val = config_target_speeds[lower_ind]
                upper_val = config_target_speeds[upper_ind]
                lower_weight = (upper_val - target_speed) / (upper_val - lower_val)
                upper_weight = (target_speed - lower_val) / (upper_val - lower_val)
                label[lower_ind] = lower_weight
                label[upper_ind] = upper_weight
        return label

    def get_indices_speed_angle(self, target_speed, brake, angle):
        target_speed_index = np.digitize(x=target_speed, bins=self.target_speed_bins)

        # Define the first index to be the brake action
        if brake:
            target_speed_index = 0
        else:
            target_speed_index += 1

        angle_index = np.digitize(x=angle, bins=self.angle_bins)

        return target_speed_index, angle_index

    def smooth_path(self, route):
        _, indices = np.unique(route, return_index=True, axis=0)
        # We need to remove the sorting of unique, because this algorithm assumes the order of the path is kept
        route = route[np.sort(indices)]
        interpolated_route_points = self.iterative_line_interpolation(route)

        return interpolated_route_points

    def iterative_line_interpolation(self, route):
        interpolated_route_points = []

        # this value is actually not used anymore, it is overwritten in the loop
        min_distance = self.config.dense_route_planner_min_distance
        target_first_distance = 2.5
        last_interpolated_point = np.array([0.0, 0.0])
        current_route_index = 0
        current_point = route[current_route_index]
        last_point = np.array([0.0, 0.0])
        first_iteration = True

        while len(interpolated_route_points) < self.config.num_route_points:
            # First point should be target_first_distance away from the vehicle.
            if not first_iteration:
                current_route_index += 1
                last_point = current_point

            if current_route_index < route.shape[0]:
                current_point = route[current_route_index]
                intersection = t_u.circle_line_segment_intersection(
                    circle_center=last_interpolated_point,
                    circle_radius=(
                        min_distance if not first_iteration else target_first_distance
                    ),
                    pt1=last_interpolated_point,
                    pt2=current_point,
                    full_line=True,
                )

            else:  # We hit the end of the input route. We extrapolate the last 2 points
                current_point = route[-1]
                last_point = route[-2]
                intersection = t_u.circle_line_segment_intersection(
                    circle_center=last_interpolated_point,
                    circle_radius=min_distance,
                    pt1=last_point,
                    pt2=current_point,
                    full_line=True,
                )

            # 3 cases: 0 intersection, 1 intersection, 2 intersection
            if len(intersection) > 1:  # 2 intersections
                # Take the one that is closer to current point
                point_1 = np.array(intersection[0])
                point_2 = np.array(intersection[1])
                direction = current_point - last_point
                dot_p1_to_last = np.dot(point_1, direction)
                dot_p2_to_last = np.dot(point_2, direction)

                if dot_p1_to_last > dot_p2_to_last:
                    intersection_point = point_1
                else:
                    intersection_point = point_2
                add_point = True
            elif len(intersection) == 1:  # 1 Intersections
                intersection_point = np.array(intersection[0])
                add_point = True
            else:  # 0 Intersection
                add_point = False
                raise Exception("No intersection found. This should never occur.")

            if add_point:
                last_interpolated_point = intersection_point
                interpolated_route_points.append(intersection_point)
                min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

            first_iteration = False

        interpolated_route_points = np.array(interpolated_route_points)
        return interpolated_route_points

    def _load_cached(self, path: Union[str, os.PathLike], load_func):
        item_i = None
        # load from cache, if enabled and found
        if self.data_cache is not None:
            item_i = self.data_cache.get(path)
        if item_i is None:
            item_i = load_func(path)
            # save to cache, if enabled
            if self.data_cache is not None:
                self.data_cache.set(key=path, value=item_i)
        return item_i

    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _load_lidar(self, path: Union[str, os.PathLike]) -> np.ndarray:
        "Loads lidar data"
        def _load(path):
            las_object = laspy.read(path)
            lidars_i = las_object.xyz
            return lidars_i

        return self._load_cached(path, _load)

    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _load_jpg(self, path: Union[str, os.PathLike]) -> np.ndarray:
        "Used for loading images"
        def _load(path):
            image_i = cv2.imread(path, cv2.IMREAD_COLOR)
            image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB)
            image_i = t_u.crop_array(self.config, image_i)
            return image_i

        return self._load_cached(path, _load)

    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _load_png(self, path: Union[str, os.PathLike], crop: bool = True) -> np.ndarray:
        "Used for loading semantics, bev_semantics and depth"
        def _load(path):
            image_i = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if crop:
                image_i = t_u.crop_array(self.config, image_i)
            return image_i

        return self._load_cached(path, _load)

    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _load_json_gz(self, path: Union[str, os.PathLike]) -> np.ndarray:
        def _load(path):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                json = ujson.load(f)
            return json
        return self._load_cached(path, _load)


def image_augmenter(prob=0.2, cutout=False):
    augmentations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(
            prob,
            ia.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        ),
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]

    if cutout:
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

    augmenter = ia.Sequential(augmentations, random_order=True)

    return augmenter


def lidar_augmenter(prob=0.2, cutout=False):
    augmentations = []

    if cutout:
        augmentations.append(
            ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False, cval=0.0))
        )

    augmenter = ia.Sequential(augmentations, random_order=True)

    return augmenter


if __name__ == "__main__":
    from config import GlobalConfig

    use_heuristic = True
    config = GlobalConfig()
    config.augment_percentage = 1
    config.color_aug_prob = 1
    # Recurrent dataset
    config.seq_len = 1
    config.seq_step = 1
    # Trajectories
    config.trajectory_step_size = 1
    config.use_trajectory_prediction = True
    config.trajectory_pred_len = 8
    # Img + lidar seq
    config.img_seq_len = 1
    config.lidar_seq_len = 1
    config.img_step_size = 1
    config.lidar_step_size = 1
    config.estimate_class_distributions = False
    config.estimate_semantic_distribution = False
    config.initialize(
        root_dir=[
            "../results/data/garage_v2_2025_03_15/data"
        ]
    )
    dataset = CARLA_Data(
        root=config.data_roots,
        config=config,
        estimate_class_distributions=config.estimate_class_distributions,
        estimate_sem_distribution=config.estimate_semantic_distribution,
        heuristic_pruning=use_heuristic
    )
    # for data in tqdm(dataset):
    #     ...
    sample = dataset.__getitem__(90)
    # print("\nShapes:")
    # for k, v in sample.items():
    #     if isinstance(v, np.ndarray):
    #         if v.shape == np.array(0).shape:
    #             print(f"'{k}': {v} ({v.dtype})")
    #         else:
    #             print(f"'{k}': shape[{v.shape}]")
    # print()
    print(sample.keys())

