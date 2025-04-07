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
    ) -> None:
        self.config = config
        self.validation = validation

        self.data_cache = shared_dict
        self.target_speed_bins = np.array(config.target_speed_bins)
        self.angle_bins = np.array(config.angle_bins)
        self.converter = np.uint8(config.converter)
        self.bev_converter = np.uint8(config.bev_converter)

        self.images = []
        self.images_augmented = []
        self.temporal_images = []
        self.semantics = []
        self.semantics_augmented = []
        self.bev_semantics = []
        self.bev_semantics_augmented = []
        self.depth = []
        self.depth_augmented = []
        self.lidars = []
        self.boxes = []
        self.future_boxes = []
        self.measurements = []
        self.future_trajectories = []
        self.sample_start = []

        self.temporal_lidars = []
        self.temporal_measurements = []

        self.image_augmenter_func = image_augmenter(
            config.color_aug_prob, cutout=config.use_cutout
        )
        self.lidar_augmenter_func = lidar_augmenter(
            config.lidar_aug_prob, cutout=config.use_cutout
        )

        # Initialize with 1 example per class
        self.angle_distribution = np.arange(len(config.angles)).tolist()
        self.speed_distribution = np.arange(len(config.target_speeds)).tolist()
        self.semantic_distribution = np.arange(len(config.semantic_weights)).tolist()
        total_routes = 0
        trainable_routes = 0
        skipped_routes = 0

        # loops over the scenarios given in root (which is a list of the scenario folders)
        for sub_root in tqdm(root, file=sys.stdout, disable=rank != 0):

            # list subdirectories in root
            routes = next(os.walk(sub_root))[1]

            for (
                route
            ) in routes:  # loop over individual routes within this scenario folder
                total_routes += 1
                route_dir = sub_root + "/" + route
                lidar_dir = route_dir + "/lidar"

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

                # If we are using checkpoints to predict the path, we can use all of the frames, otherwise we need to subtract
                # pred_len so that we have enough waypoint labels
                # last_frame = (
                #     num_seq
                #     - (self.config.seq_len - 1) * self.config.seq_step
                #     - (0 if not self.config.use_wp_gru else self.config.pred_len)
                # )

                # Subtract maximum of the forcasting times
                last_frame = num_seq - max(
                    0, 
                    (self.config.seq_len - 1) * self.config.seq_step, 
                    (
                        0 if not self.config.use_trajectory_prediction else 
                        1 + (self.config.trajectory_pred_len) * self.config.trajectory_step_size
                    ),
                    (0 if not self.config.use_wp_gru else self.config.pred_len),
                )

                if last_frame <= first_frame:
                    warnings.warn(f"Not enough frames in {route_dir} for given sequence length (rgb: {config.img_seq_len}, lidar: {config.lidar_seq_len}, all: {config.seq_len}) and step size (rgb: {config.img_step_size}, lidar: {config.lidar_step_size}, all: {config.seq_step}), skipping route.")
                    skipped_routes += 1
                    continue

                trainable_routes += 1
                # For all frames of the route (that we want to load)
                for seq in range(first_frame, last_frame):
                    if seq % config.train_sampling_rate != 0:
                        continue

                    # load input seq and pred seq jointly
                    image = []
                    image_augmented = []
                    semantic = []
                    semantic_augmented = []
                    bev_semantic = []
                    bev_semantic_augmented = []
                    depth = []
                    depth_augmented = []
                    lidar = []
                    box = []
                    future_box = []

                    # we only store the root and compute the file name when loading,
                    # because storing 40 * long string per sample can go out of memory.
                    measurement = route_dir + "/measurements"
                    future_trajectories = route_dir + "/boxes"

                    # Load sequence of frames (according to config.seq_len)
                    for idx in range(
                        0,
                        self.config.seq_len * self.config.seq_step,
                        self.config.seq_step,
                    ):

                        image.append(route_dir + "/rgb" + (f"/{(seq + idx):04}.jpg"))
                        image_augmented.append(
                            route_dir + "/rgb_augmented" + (f"/{(seq + idx):04}.jpg")
                        )
                        semantic.append(
                            route_dir + "/semantics" + (f"/{(seq + idx):04}.png")
                        )
                        semantic_augmented.append(
                            route_dir
                            + "/semantics_augmented"
                            + (f"/{(seq + idx):04}.png")
                        )
                        bev_semantic.append(
                            route_dir + "/bev_semantics" + (f"/{(seq + idx):04}.png")
                        )
                        bev_semantic_augmented.append(
                            route_dir
                            + "/bev_semantics_augmented"
                            + (f"/{(seq + idx):04}.png")
                        )
                        depth.append(route_dir + "/depth" + (f"/{(seq + idx):04}.png"))
                        depth_augmented.append(
                            route_dir + "/depth_augmented" + (f"/{(seq + idx):04}.png")
                        )
                        lidar.append(route_dir + "/lidar" + (f"/{(seq + idx):04}.laz"))

                        if estimate_sem_distribution:
                            semantics_i = self.converter[
                                self._load_png(semantic[-1], crop=False)
                            ]  # pylint: disable=locally-disabled, unsubscriptable-object
                            self.semantic_distribution.extend(
                                semantics_i.flatten().tolist()
                            )

                        forcast_step = int(
                            config.forcast_time
                            / (config.data_save_freq / config.carla_fps)
                            + 0.5
                        )

                        box.append(
                            route_dir + "/boxes" + (f"/{(seq + idx):04}.json.gz")
                        )
                        future_box.append(
                            route_dir
                            + "/boxes"
                            + (f"/{(seq + idx + forcast_step):04}.json.gz")
                        )
                        # measurement.append(
                        #     route_dir +
                        #     "/measurements"
                        #     + f"/{(seq + idx + forcast_step):04}.json.gz"
                        # )



                    if estimate_class_distributions:
                        measurements_i = self._load_json_gz(
                            measurement[-1] + f"/{(seq):04}.json.gz"
                        )

                        target_speed_index, angle_index = self.get_indices_speed_angle(
                            target_speed=measurements_i["target_speed"],
                            brake=measurements_i["brake"],
                            angle=measurements_i["angle"],
                        )

                        self.angle_distribution.append(angle_index)
                        self.speed_distribution.append(target_speed_index)
                    if self.config.lidar_seq_len > 1:
                        assert self.config.seq_len == 1
                        # load input seq and pred seq jointly
                        temporal_lidar = []
                        temporal_measurement = []
                        for idx in range(
                            0,
                            self.config.lidar_seq_len * config.lidar_step_size,
                            config.lidar_step_size,
                        ):
                            # Temporal LiDARs are only supported with seq len 1 right now
                            temporal_lidar.append(
                                route_dir + "/lidar" + (f"/{(seq - idx):04}.laz")
                            )
                            temporal_measurement.append(
                                route_dir
                                + "/measurements"
                                + (f"/{(seq - idx):04}.json.gz")
                            )

                        self.temporal_lidars.append(temporal_lidar)
                        self.temporal_measurements.append(temporal_measurement)

                    # Add temporal img index
                    if self.config.img_seq_len > 1:
                        assert self.config.seq_len == 1
                        self.temporal_images.append(
                            [
                                route_dir + "/rgb" + (f"/{(seq - idx):04}.jpg")
                                for idx in range(
                                    0,
                                    self.config.img_seq_len * config.img_step_size,
                                    config.img_step_size,
                                )
                            ]
                        )

                    self.images.append(image)
                    self.images_augmented.append(image_augmented)
                    self.semantics.append(semantic)
                    self.semantics_augmented.append(semantic_augmented)
                    self.bev_semantics.append(bev_semantic)
                    self.bev_semantics_augmented.append(bev_semantic_augmented)
                    self.depth.append(depth)
                    self.depth_augmented.append(depth_augmented)
                    self.lidars.append(lidar)
                    self.boxes.append(box)
                    self.future_boxes.append(future_box)
                    self.measurements.append(measurement)
                    self.future_trajectories.append(future_trajectories)
                    self.sample_start.append(seq)

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
        self.images = np.array(self.images).astype(np.string_)
        self.images_augmented = np.array(self.images_augmented).astype(np.string_)
        self.temporal_images = np.array(self.temporal_images).astype(np.string_)
        self.semantics = np.array(self.semantics).astype(np.string_)
        self.semantics_augmented = np.array(self.semantics_augmented).astype(np.string_)
        self.bev_semantics = np.array(self.bev_semantics).astype(np.string_)
        self.bev_semantics_augmented = np.array(self.bev_semantics_augmented).astype(
            np.string_
        )
        self.depth = np.array(self.depth).astype(np.string_)
        self.depth_augmented = np.array(self.depth_augmented).astype(np.string_)
        self.lidars = np.array(self.lidars).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.future_boxes = np.array(self.future_boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        self.future_trajectories = np.array(self.future_trajectories).astype(np.string_)

        self.temporal_lidars = np.array(self.temporal_lidars).astype(np.string_)
        self.temporal_measurements = np.array(self.temporal_measurements).astype(
            np.string_
        )
        self.sample_start = np.array(self.sample_start)
        if rank == 0:
            print(f"Loading {len(self.lidars)} lidars from {len(root)} folders")
            print("Total amount of routes:", total_routes)
            print("Skipped routes:", skipped_routes)
            print("Trainable routes:", trainable_routes)

    def __len__(self):
        """Returns the length of the dataset."""
        return self.lidars.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx."""
        # Disable threading because the data loader will already split in processes.
        cv2.setNumThreads(0)

        data = {}

        images = self.images[index]
        images_augmented = self.images_augmented[index]
        semantics = self.semantics[index]
        semantics_augmented = self.semantics_augmented[index]
        bev_semantics = self.bev_semantics[index]
        bev_semantics_augmented = self.bev_semantics_augmented[index]
        depth = self.depth[index]
        depth_augmented = self.depth_augmented[index]
        lidars = self.lidars[index]
        boxes = self.boxes[index]
        future_boxes = self.future_boxes[index]

        if self.config.img_seq_len > 1:
            assert (
                self.config.seq_len == 1
            ), "img_seq_len>1 can only be used with seq_len=1"
            temporal_images = self.temporal_images[index]
        if self.config.lidar_seq_len > 1:
            assert (
                self.config.seq_len == 1
            ), "lidar_seq_len>1 can only be used with seq_len=1"
            temporal_lidars = self.temporal_lidars[index]
            temporal_measurements = self.temporal_measurements[index]

        # we need to calculate the paths, since they are too large to put in the index
        measurement_root = self.measurements[index]
        trajectories_root = self.future_trajectories[index]
        sample_start = self.sample_start[index]

        # Since we load measurements for future time steps, we load and store them separately
        loaded_measurements = []
        for i in range(self.config.seq_len):
            measurement_file = str(measurement_root, encoding="utf-8") + (
                f"/{(sample_start + i*self.config.seq_step):04}.json.gz"
            )
            measurements_i = self._load_json_gz(measurement_file)
            loaded_measurements.append(measurements_i)

        # For lidar alignment, we need the current frame
        # TODO: but what is "current" if temporal frames are processed independently?
        current_measurement = loaded_measurements[self.config.seq_len - 1]  # last

        # Same for trajectories
        loaded_trajectories = []
        if self.config.use_trajectory_prediction:
            # Load future trajectory data
            for i in range(self.config.trajectory_pred_len):
                trajectory_file = str(trajectories_root, encoding="utf-8") + (
                    f"/{((sample_start+self.config.seq_len-1) + (i)*self.config.trajectory_step_size):04}.json.gz"
                )
                temporal_boxes_i = self._load_json_gz(trajectory_file)
                loaded_trajectories.append(temporal_boxes_i)

        # If we were to use the GRU for WP prediction, we need the future measurements
        # (is how I interpret this code at least)
        if self.config.use_wp_gru:
            end = self.config.pred_len + self.config.seq_len
            start = self.config.seq_len
            for i in range(start, end, self.config.wp_dilation):
                measurement_file = str(measurement_root, encoding="utf-8") + (
                    f"/{(sample_start + i*self.config.seq_step):04}.json.gz"
                )
                measurements_i = self._load_json_gz(measurement_file)
                loaded_measurements.append(measurements_i)

        loaded_temporal_lidars = []
        loaded_temporal_measurements = []
        if self.config.lidar_seq_len > 1:
            # Temporal data just for LiDAR
            for i in range(self.config.lidar_seq_len):
                temporal_measurements_i = self._load_json_gz(temporal_measurements[i])
                temporal_lidars_i = self._load_lidar(
                    str(temporal_lidars[i], encoding="utf-8")
                )

                loaded_temporal_lidars.append(temporal_lidars_i)
                loaded_temporal_measurements.append(temporal_measurements_i)

            loaded_temporal_lidars.reverse()
            loaded_temporal_measurements.reverse()

        # Here we load all inputs as sequences of frames
        # This would allow a model to process the data sequentially if seq_len > 1
        # To avoid complications, temporal rgb and temporal lidar are loaded in a separate loop
        # Temporal rgb and temporal lidar should not be used with seq_len > 1
        #   (it makes no sense to have a sequence of sequences of images or lidars...)

        loaded_images = []
        loaded_semantics = []
        loaded_bev_semantics = []
        loaded_depth = []
        loaded_lidars = []
        loaded_boxes = []
        loaded_future_boxes = []
        # bounding box related data
        box_targets = []
        avg_factors = []

        target_point_seq = []
        target_point_next_seq = []
        route_seq = []

        brake_seq = []
        angle_index_seq = []
        target_speed_seq = []
        target_speed_twohot_seq = []

        # TODO:
        #   maybe ralignment should be turned off if we use a sequence?
        #   It kinda only makes sense if we input the whole sequence to the model
        #   Like early temporal fusion
        #   With temporal streaming, the frames would realign to an unknown future frame, which doesn't make sense
        #   Perhaps this should be discussed in more depth

        # Load and process all frames
        for idx in range(self.config.seq_len):
            measurement_i = loaded_measurements[idx]

            # Determine whether the augmented camera or the normal camera is used.
            if (
                random.random() <= self.config.augment_percentage
                and self.config.augment
            ):
                # TODO: should we use per-timestep augment, or one for all timesteps?
                aug_rotation = measurement_i["augmentation_rotation"]
                aug_translation = measurement_i["augmentation_translation"]
                images_path = str(images_augmented[idx], encoding="utf-8")
                semantics_path = str(semantics_augmented[idx], encoding="utf-8")
                bev_semantics_path = str(bev_semantics_augmented[idx], encoding="utf-8")
                depth_path = str(depth_augmented[idx], encoding="utf-8")
            else:
                aug_rotation = 0.0
                aug_translation = 0.0
                images_path = str(images[idx], encoding="utf-8")
                semantics_path = str(semantics[idx], encoding="utf-8")
                bev_semantics_path = str(bev_semantics[idx], encoding="utf-8")
                depth_path = str(depth[idx], encoding="utf-8")

            # Augment target points
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

            # Augment and process the route
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

            # Convert target speed and angles to indexes
            brake = measurement_i["brake"]

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
            brake_seq.append(brake)

            # Because the strings are stored as numpy byte objects we need to
            # convert them back to utf-8 strings

            image_i = self._process_image(self._load_jpg(images_path))
            loaded_images.append(image_i)

            # Load optional aux images
            if self.config.use_semantic:
                semantic_i = self._process_semantics(self._load_png(semantics_path))
                loaded_semantics.append(semantic_i)
            if self.config.use_bev_semantic:
                bev_semantic_i = self._process_bev_semantics(
                    self._load_png(bev_semantics_path, crop=False)
                )
                loaded_bev_semantics.append(bev_semantic_i)
            if self.config.use_depth:
                depth_i = self._process_depth(self._load_png(depth_path))
                loaded_depth.append(depth_i)

            # Load boxes (and future boxes)
            if self.config.detect_boxes:
                boxes_i = future_boxes_i = None

                box_path = str(boxes[idx], encoding="utf-8")
                boxes_i = self._load_json_gz(box_path)

                if self.config.use_plant:
                    future_box_path = str(future_boxes[idx], encoding="utf-8")
                    future_boxes_i = self._load_json_gz(future_box_path)

                # Process and pad the boxes
                boxes_i, boxes_padded_i, _, future_boxes_padded_i = self._process_boxes(
                    boxes_i, future_boxes_i, aug_translation, aug_rotation
                )
                loaded_boxes.append(boxes_padded_i)
                if future_boxes_i is not None:
                    loaded_future_boxes.append(future_boxes_padded_i)

                # Get the targets from the current boxes
                target_result, avg_factor = self.get_targets(
                    boxes_i,
                    self.config.lidar_resolution_height
                    // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width
                    // self.config.bev_down_sample_factor,
                )
                box_targets.append(target_result)
                avg_factors.append(avg_factor)

            # Load and align lidar
            # need to concatenate seq data here and align to the same coordinate
            lidar_path = str(lidars[idx], encoding="utf-8")
            lidar = self._load_lidar(lidar_path)
            # transform lidar to lidar seq-1
            lidar = self.align(  # TODO: should we align to each time step?
                lidar,
                measurements_i,  # if not realign, only applies augments
                current_measurement if self.config.realign_lidar else measurements_i,
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )
            lidar = self.lidar_to_histogram_features(
                lidar, use_ground_plane=self.config.use_ground_plane
            )
            lidar = self.lidar_augmenter_func(image=np.transpose(lidar, (1, 2, 0)))
            lidar = np.transpose(lidar, (2, 0, 1))
            loaded_lidars.append(lidar)

        # Converting to array retains the seq dimension
        # For seq=1, this dimension is removed later
        data["rgb"] = np.array(loaded_images)
        data["lidar"] = np.array(loaded_lidars)
        data["target_point"] = np.array(target_point_seq)
        data["target_point_next"] = np.array(target_point_next_seq)
        data["route"] = np.array(route_seq)
        data["brake"] = np.array(brake_seq)
        data["angle_index"] = np.array(angle_index_seq)
        data["target_speed"] = np.array(target_speed_seq)
        data["target_speed_twohot"] = np.array(target_speed_twohot_seq)
        # Optional stuff
        if self.config.use_semantic:
            data["semantic"] = np.array(loaded_semantics)
        if self.config.use_bev_semantic:
            data["bev_semantic"] = np.array(loaded_bev_semantics)
        if self.config.use_depth:
            data["depth"] = np.array(loaded_depth)
        if self.config.detect_boxes:
            data["bounding_boxes"] = np.array(loaded_boxes)
            if self.config.use_plant:
                data["future_bounding_boxes"] = np.array(loaded_future_boxes)
            data["center_heatmap"] = np.array(
                [t["center_heatmap_target"] for t in box_targets]
            )
            data["wh"] = np.array([t["wh_target"] for t in box_targets])
            data["yaw_class"] = np.array([t["yaw_class_target"] for t in box_targets])
            data["yaw_res"] = np.array([t["yaw_res_target"] for t in box_targets])
            data["offset"] = np.array([t["offset_target"] for t in box_targets])
            data["velocity"] = np.array([t["velocity_target"] for t in box_targets])
            data["brake_target"] = np.array([t["brake_target"] for t in box_targets])
            data["pixel_weight"] = np.array([t["pixel_weight"] for t in box_targets])
            data["avg_factor"] = np.array(avg_factors)

        # The rest can be handled by this simple one-liner
        #   dict of lists, like what a data loader does to batches
        measurement_seq = loaded_measurements[: self.config.seq_len]
        meas_dict_seq = {
            k: np.array(
                [
                    d[k] for d in measurement_seq
                ]  # val is list of vals of all dicts at that key
            )
            for k in measurement_seq[0]  # keys of sample dict (assumes all same keys)
        }
        data["steer"] = meas_dict_seq["steer"]
        data["throttle"] = meas_dict_seq["throttle"]
        data["light"] = meas_dict_seq["light_hazard"]
        data["stop_sign"] = meas_dict_seq["stop_sign_hazard"]
        data["junction"] = meas_dict_seq["junction"]
        data["speed"] = meas_dict_seq["speed"]
        data["theta"] = meas_dict_seq["theta"]
        # Command one hot list comprehension
        data["command"] = np.array(
            [t_u.command_to_one_hot(cmd) for cmd in meas_dict_seq["command"]]
        )
        data["next_command"] = np.array(
            [t_u.command_to_one_hot(cmd) for cmd in meas_dict_seq["next_command"]]
        )

        # Load temporal images
        loaded_temporal_images = []
        if self.config.img_seq_len > 1:
            # TODO: does it make sense to apply agumentations here?
            # Temporal data just for LiDAR
            for i in range(self.config.img_seq_len):
                img_path = str(temporal_images[i], encoding="utf-8")
                temporal_image_i = self._process_image(self._load_jpg(img_path))
                loaded_temporal_images.append(temporal_image_i)

            loaded_temporal_images.reverse()
            data["temporal_rgb"] = np.array(loaded_temporal_images)

        # Load temporal lidars
        if self.config.lidar_seq_len > 1:
            temporal_lidars = []
            for i in range(self.config.lidar_seq_len):
                # transform lidar to lidar seq-1
                if self.config.realign_lidar:
                    temporal_lidar = self.align(
                        loaded_temporal_lidars[i],
                        loaded_temporal_measurements[i],
                        loaded_temporal_measurements[self.config.lidar_seq_len - 1],
                        # This is cheat, the augs are retained outside the local scope (because python)
                        # We know that this is the correct aug since we only have one aug with lidar_seq_len > 1.
                        # It is not pretty, however, and should be fixed (TODO)
                        y_augmentation=aug_translation,
                        yaw_augmentation=aug_rotation,
                    )
                else:
                    # For data augmentation to still occur.
                    temporal_lidar = self.align(
                        loaded_temporal_lidars[i],
                        loaded_temporal_measurements[i],
                        loaded_temporal_measurements[i],
                        # Same here (TODO)
                        y_augmentation=aug_translation,
                        yaw_augmentation=aug_rotation,
                    )
                temporal_lidar = self.lidar_to_histogram_features(
                    temporal_lidar, use_ground_plane=self.config.use_ground_plane
                )
                temporal_lidars.append(temporal_lidar)

            temporal_lidar_bev = np.concatenate(temporal_lidars, axis=0)
            temporal_lidar_bev = self.lidar_augmenter_func(
                image=np.transpose(temporal_lidar_bev, (1, 2, 0))
            )
            data["temporal_lidar"] = np.transpose(temporal_lidar_bev, (2, 0, 1))

        if self.config.use_trajectory_prediction:
            # TODO, this is a bit stupid, since we could get it from above
            box_path = str(boxes[self.config.seq_len - 1], encoding="utf-8")
            current_boxes = self._load_json_gz(box_path)

            traj, mask = self._process_trajectories(
                loaded_trajectories,
                current_boxes=current_boxes,
                current_measurement=current_measurement,
                y_augmentation=aug_translation,
                yaw_augmentation=aug_rotation,
            )
            data["trajectories"] = traj
            data["trajectories_mask"] = mask

        # if self.config.use_wp_gru:
        #     # TODO we can ignore this
        #     waypoints = self.get_waypoints(
        #         loaded_measurements[self.config.seq_len - 1 :],
        #         y_augmentation=aug_translation,
        #         yaw_augmentation=aug_rotation,
        #     )

        #     data["ego_waypoints"] = np.array(waypoints)

        # Finally, if seq==1, remove the seq dimension
        if self.config.seq_len == 1:
            data = {
                k: v.squeeze(0) if k not in ["temporal_lidar", "temporal_rgb", "trajectories", "trajectories_mask"] else v
                for k, v in data.items()
            }

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

    def _process_trajectories(self, temporal_boxes, current_boxes, current_measurement, y_augmentation=0.0, yaw_augmentation=0.0):
        
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
        for boxes in temporal_boxes:  # [F, n, ...]

            # Parse the boxes
            parsed_boxes, orig_boxes = self.parse_bounding_boxes_traj(
                future_boxes=boxes,  # current timestep future
                reference=current_boxes,
                y_augmentation=y_augmentation,
                yaw_augmentation=yaw_augmentation
            )

            # For all actors in this timestep, get their id and their relative position
            id_to_xy = {}
            for box_data, box in zip(parsed_boxes, orig_boxes):
                id_to_xy[box['id']] = box_data[:2]  # [x, y]
            all_timesteps.append(id_to_xy)

        # All actors present in *some* timestep:
        # all_ids = list(set([k for d in all_timesteps for k in d.keys()]))
        
        # all_ids = list(set(all_timesteps[0]))  # only keys appearing in t=0

        # Create a mask for valid trajectories,
        # some actors enter and leave the bounds during the timesteps, these should be masked
        mask = np.zeros((F, N))

        # For all actors, for all timesteps, get the position of that actor at that timestep
        # If position is found, append position and set mask to 1
        # Else, append zero and set mask to 0
        all_trajectories = []  # should be [n, F, 2]
        for actor_i, actor in enumerate(valid_ids[: N]): # [min(n,N)], truncate if >N
            actor_traj = []
            for t, d in enumerate(all_timesteps):  # [F, n]
                pos = d.get(actor, None)
                if pos is not None:
                    actor_traj.append(pos)
                    mask[t, actor_i] = 1
                else:
                    actor_traj.append(np.zeros((2)))
            all_trajectories.append(actor_traj)

        # These are now Ego_0-aligned trajectories for all actors visible in any timestep
        all_trajectories = np.array(all_trajectories)  # [n, F, 2]
        mask = np.transpose(mask, (1,0))  # [F, n] -> [n, F]

        # TODO: filter cars with mask=0 on first timestep

        # Pad all trajectories, we want [n, F, 2] to [N, F, 2]
        n = all_trajectories.shape[0]  # num actors
        all_trajectories_padded = np.zeros((N, F, 2))
        all_trajectories_padded[: min(n,N)] = all_trajectories[: min(n,N)]  

        return all_trajectories_padded, mask

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

        # current_ids = [b['id'] for b in reference]
        valid_current_ids = []
        for box in reference:
            bbox, height = self.get_bbox_label(
                box, y_augmentation, yaw_augmentation
            )
            if not (
                bbox[0] <= self.config.min_x
                or bbox[0] >= self.config.max_x
                or bbox[1] <= self.config.min_y
                or bbox[1] >= self.config.max_y
                or height <= self.config.min_z
                or height >= self.config.max_z
            ):
                valid_current_ids.append(box['id'])

        bboxes = []
        original_boxes = []
        for idx, sample_box in enumerate(future_boxes):
            # (1) Filter out non-valid boxes
            # Should we filter out cars not visible in the current frame?
            condition0 = True#sample_box["id"] in valid_current_ids
        
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
            if not (condition0 and condition1 and condition2):
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

    def _load_lidar(self, path: Union[str, os.PathLike]) -> np.ndarray:
        "Loads lidar data"
        def _load(path):
            las_object = laspy.read(path)
            lidars_i = las_object.xyz
            return lidars_i
        
        return self._load_cached(path, _load)

    def _load_jpg(self, path: Union[str, os.PathLike]) -> np.ndarray:
        "Used for loading images"
        def _load(path):
            image_i = cv2.imread(path, cv2.IMREAD_COLOR)
            image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB)
            image_i = t_u.crop_array(self.config, image_i)
            return image_i
        
        return self._load_cached(path, _load)

    def _load_png(self, path: Union[str, os.PathLike], crop: bool = True) -> np.ndarray:
        "Used for loading semantics, bev_semantics and depth"
        def _load(path):
            image_i = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if crop:
                image_i = t_u.crop_array(self.config, image_i)
            return image_i

        return self._load_cached(path, _load)

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

    config = GlobalConfig()
    config.seq_len = 1
    config.lidar_seq_len = 1
    config.img_seq_len = 1
    config.use_trajectory_prediction = True
    config.initialize(
        root_dir=[
            "/cluster/work/andrebw/repos/temporal_garage/results/data/garage_v2_2025_03_15/data"
        ]
    )
    dataset = CARLA_Data(
        root=config.data_roots,
        config=config,
        estimate_class_distributions=config.estimate_class_distributions,
        estimate_sem_distribution=config.estimate_semantic_distribution,
    )
    sample = dataset.__getitem__(501)
    print("\nShapes:")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            if v.shape == np.array(0).shape:
                print(f"'{k}': {v} ({v.dtype})")
            else:
                print(f"'{k}': shape[{v.shape}]")
    print()
    print(sample.keys())
