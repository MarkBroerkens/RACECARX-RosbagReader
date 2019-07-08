# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import imghdr
import argparse
import functools
import numpy as np
import pandas as pd

from bagutils import *


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def check_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % image_filename)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            # Avoid re-encoding if we don't have to
            if check_format(msg.data) == fmt:
                buf.tofile(image_filename)
            else:
                cv2.imwrite(image_filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    results['filename'] = image_filename
    return results


def camera2dict(msg, write_results, camera_dict):
    camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
    camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
    camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
    camera_dict["frame_id"].append(msg.header.frame_id)
    camera_dict["filename"].append(write_results['filename'])


def imu2dict(msg, imu_dict):
    imu_dict["timestamp"].append(msg.header.stamp.to_nsec())
    imu_dict["ax"].append(msg.linear_acceleration.x)
    imu_dict["ay"].append(msg.linear_acceleration.y)
    imu_dict["az"].append(msg.linear_acceleration.z)


def low_level_ackermann2dict(msg, low_level_ackermann_dict):
    low_level_ackermann_dict["timestamp"].append(msg.header.stamp.to_nsec())
    low_level_ackermann_dict["steering_angle"].append(msg.drive.steering_angle)
    low_level_ackermann_dict['steering_angle_velocity'].append(msg.drive.steering_angle_velocity)
    low_level_ackermann_dict["speed"].append(msg.drive.speed)
    low_level_ackermann_dict["acceleration"].append(msg.drive.acceleration)
    low_level_ackermann_dict["jerk"].append(msg.drive.jerk)


def odom2dict(msg, odom_dict):
    odom_dict["timestamp"].append(msg.header.stamp.to_nsec())
    odom_dict["vx"].append(msg.twist.twist.linear.x)
    odom_dict['vy'].append(msg.twist.twist.linear.y)


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where bagfiles are located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    msg_only = args.msg_only
    debug_print = args.debug

    bridge = CvBridge()

    include_images = False if msg_only else True
    include_others = True

    filter_topics = [LOW_LEVEL_ACKERMANN_TOPIC, ODOMETRY_TOPIC]
    if include_images:
        filter_topics += CAMERA_TOPICS
    if include_others:
        filter_topics += OTHER_TOPICS

    bagsets = find_bagsets(indir, filter_topics=filter_topics)
    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        center_outdir = get_outdir(dataset_outdir, "center")

        camera_cols = ["timestamp", "width", "height", "frame_id", "filename"] 
        camera_dict = defaultdict(list)

        low_level_ackermann_cols = ["timestamp", "steering_angle", "steering_angle_velocity", "speed", "acceleration", "jerk"]
        low_level_ackermann_dict = defaultdict(list)

        odom_cols = ["timestamp", "vx", "vy"]
        odom_dict = defaultdict(list)

        if include_others:
            imu_cols = ["timestamp", "ax", "ay", "az"]
            imu_dict = defaultdict(list)

        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, stats):
            timestamp = msg.header.stamp.to_nsec()
            if topic in CAMERA_TOPICS:
                outdir = center_outdir
                if debug_print:
                    print("%s_camera %d" % (topic[1], timestamp))

                results = write_image(bridge, outdir, msg, fmt=img_format)
                results['filename'] = os.path.relpath(results['filename'], dataset_outdir)
                camera2dict(msg, results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic == LOW_LEVEL_ACKERMANN_TOPIC:
                if debug_print:
                    print("low level ackermann %d %f" % (timestamp, msg.drive.steering_angle))

                low_level_ackermann2dict(msg, low_level_ackermann_dict)
                stats['msg_count'] += 1

            elif topic == ODOMETRY_TOPIC:
                if debug_print:
                    print("odometry %d %f %f" % (timestamp, msg.twist.twist.linear.x, msg.twist.twist.linear.y))

                odom2dict(msg, odom_dict)
                stats['msg_count'] += 1



            else:
                if include_others:
                    if topic == IMU_TOPIC:
                        imu2dict(msg, imu_dict)
                        stats['msg_count'] += 1

        # no need to cycle through readers in any order for dumping, rip through each on in sequence
        for reader in readers:
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if ((stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0) or
                        (stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0)):
                    print("%d images, %d messages processed..." %
                          (stats_acc['img_count'], stats_acc['msg_count']))
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        if include_images:
            camera_csv_path = os.path.join(dataset_outdir, 'camera.csv')
            camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
            camera_df.to_csv(camera_csv_path, index=False)

        low_level_ackermann_csv_path = os.path.join(dataset_outdir, 'low_level_ackermann.csv')
        low_level_ackermann_df = pd.DataFrame(data=low_level_ackermann_dict, columns=low_level_ackermann_cols)
        low_level_ackermann_df.to_csv(low_level_ackermann_csv_path, index=False)

        odom_csv_path = os.path.join(dataset_outdir, 'odom.csv')
        odom_df = pd.DataFrame(data=odom_dict, columns=odom_cols)
        odom_df.to_csv(odom_csv_path, index=False)

        if include_others:
            imu_csv_path = os.path.join(dataset_outdir, 'imu.csv')
            imu_df = pd.DataFrame(data=imu_dict, columns=imu_cols)
            imu_df.to_csv(imu_csv_path, index=False)

        gen_interpolated = True
        if include_images and gen_interpolated:
            # A little pandas magic to interpolate steering samples to camera frames
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
            camera_df.set_index(['timestamp'], inplace=True)
            camera_df.index.rename('index', inplace=True)
            low_level_ackermann_df['timestamp'] = pd.to_datetime(low_level_ackermann_df['timestamp'])
            low_level_ackermann_df.set_index(['timestamp'], inplace=True)
            low_level_ackermann_df.index.rename('index', inplace=True)
            odom_df['timestamp'] = pd.to_datetime(odom_df['timestamp'])
            odom_df.set_index(['timestamp'], inplace=True)
            odom_df.index.rename('index', inplace=True)

            merged = functools.reduce(lambda left, right: pd.merge(
                left, right, how='outer', left_index=True, right_index=True), [camera_df, low_level_ackermann_df, odom_df])
            merged.interpolate(method='time', inplace=True)

            filtered_cols = ['timestamp', 'width', 'height', 'frame_id', 'filename',
                             'steering_angle', 'speed', 'vx', 'vy']
            filtered = merged.loc[camera_df.index]  # back to only camera rows
            filtered.fillna(0.0, inplace=True)
            filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
            filtered['width'] = filtered['width'].astype('int')  # cast back to int
            filtered['height'] = filtered['height'].astype('int')  # cast back to int
            filtered = filtered[filtered_cols]  # filter and reorder columns for final output

            interpolated_csv_path = os.path.join(dataset_outdir, 'interpolated.csv')
            filtered.to_csv(interpolated_csv_path, header=True)


if __name__ == '__main__':
    main()
