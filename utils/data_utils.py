import os
import re
import cv2
import bisect
import numpy as np
import pandas as pd
from pypcd_imp import pypcd
from scipy.spatial.transform import Rotation as R
from numpy.lib.recfunctions import structured_to_unstructured

def get_timestamps(df:pd.DataFrame) -> list:
    """
    Returns all timestamps from a dataframe.
    """
    return list(10**9 * df['ts1'] + df['ts2'])

def find_nearest(x:list, q:int) -> int:
    """
    Find nearest index to a timestamp from list of timestamps.\n
    Assumes, list of timestamps is sorted in ascending order.
    """
    i = bisect.bisect(x, q)
    if i > 1 and abs(x[i-1] - q) < abs(x[i] - q):
        i = i - 1
    return i

def clean_matrices(mat:str) -> list:
    """
    Convert string to matrix.
    """
    return [[float(num) for num in row.split()] for row in mat.strip().split('\n')]

class IMAGE(object):
    """
    Class to read IMAGE timestamp files and get frames closest to a timestamp.\n
    Or simply grab a frame by frame number.
    """

    def __init__(self, path:str) -> None:
        self.filename = os.path.join(path, '1_IMAGE', 'timestamp_image_left.txt')
        data = pd.read_csv(self.filename, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2'])
        self.timestamps = get_timestamps(data)
        self.image_left = [os.path.join(path, '1_IMAGE', 'LEFT', f'{i:06d}.png') for i in range(len(data))]
        self.image_right = [os.path.join(path, '1_IMAGE', 'RIGHT', f'{i:06d}.png') for i in range(len(data))]
    
    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp and left+right images.
        """
        return self.timestamps[frame_id], cv2.imread(self.image_left[frame_id]), cv2.imread(self.image_right[frame_id])
    
    def get_frame_by_timestamp(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp and left+right images.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class LIDAR(object):
    """
    Class to read LIDAR timestamp files and get frames closest to a timestamp.\n
    Or simply grab a frame by frame number.
    """

    def __init__(self, path:str) -> None:
        self.filename = os.path.join(path, '2_LIDAR', 'timestamp_lidar.txt')
        data = pd.read_csv(self.filename, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2'])
        self.timestamps = get_timestamps(data)
        self.pcd_file = [os.path.join(path, '2_LIDAR', 'PCD', f'{i:06d}.pcd') for i in range(len(data))]
    
    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp and PCD.
        """
        pts = structured_to_unstructured(pypcd.PointCloud.from_path(self.pcd_file[frame_id]).pc_data)
        return self.timestamps[frame_id], pts
    
    def get_frame_by_timestamp(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp and PCD.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class RADAR(object):
    """
    Class to read LIDAR timestamp files and get frames closest to a timestamp.\n
    Or simply grab a frame by frame number.
    """

    def __init__(self, path:str):
        self.filename = os.path.join(path, '3_RADAR', 'timestamp_radar.txt')
        data = pd.read_csv(self.filename, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2'])
        self.timestamps = get_timestamps(data)
        self.pcd_file = [os.path.join(path, '3_RADAR', 'PCD', f'{i:06d}.pcd') for i in range(len(data))]
    
    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp and PCD.
        """
        pts = structured_to_unstructured(pypcd.PointCloud.from_path(self.pcd_file[frame_id]).pc_data)
        return self.timestamps[frame_id], pts
    
    def get_frame_by_timestamp(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp and PCD.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class IMU(object):

    def __init__(self, path:str) -> None:
        self.filename = os.path.join(path, '4_NAVIGATION', 'IMU.txt')
        data = pd.read_csv(self.filename, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2',
                                                                        'acc_x', 'acc_y', 'acc_z',
                                                                        'ang_x', 'ang_y', 'ang_z',
                                                                        'quat_x', 'quat_y', 'quat_z', 'quat_w'])
        self.timestamps = get_timestamps(data)
        self.vel = np.column_stack((data['acc_x'], data['acc_y'], data['acc_z']))
        self.acc = np.column_stack((data['ang_x'], data['ang_y'], data['ang_z']))
        self.quat = np.column_stack((data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']))

    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp, angular velocities, linear accelerations and quaternion.
        """
        return self.timestamps[frame_id], self.vel[frame_id], self.acc[frame_id], R.from_quat(self.quat[frame_id])

    def get_frame_by_timestamps(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp, angular velocities, linear accelerations and quaternion.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class RTK_GPS(object):

    def __init__(self, path:str) -> None:
        self.filename = os.path.join(path, '4_NAVIGATION', 'RTK_GPS.txt')
        data = pd.read_csv(self.filename, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2',
                                                                        'lat', 'lon', 'alt'])
        self.timestamps = get_timestamps(data)
        self.gps = np.column_stack((data['lat'], data['lon'], data['alt']))
    
    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp and GPS position.
        """
        return self.timestamps[frame_id], self.gps[frame_id]
    
    def get_frame_by_timestamps(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp and GPS position.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class GPS(object):

    def __init__(self, path:str) -> None:
        self.filename = os.path.join(path, '4_NAVIGATION', 'GPS.txt')
        data = pd.read_csv(self.filename, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2',
                                                                        'lat', 'lon', 'alt'])
        self.timestamps = get_timestamps(data)
        self.gps = np.column_stack((data['lat'], data['lon'], data['alt']))
    
    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp and GPS position.
        """
        return self.timestamps[frame_id], self.gps[frame_id]
    
    def get_frame_by_timestamps(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp and GPS position.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class WHEEL(object):

    def __init__(self, path:str) -> None:
        self.wheel_left = os.path.join(path, '4_NAVIGATION', 'WHEEL_LEFT.txt')
        self.wheel_right = os.path.join(path, '4_NAVIGATION', 'WHEEL_RIGHT.txt')
        data_left = pd.read_csv(self.wheel_left, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2',
                                                                        'lin_vel_left', 'ang_vel_left'])
        data_right = pd.read_csv(self.wheel_left, sep=' ', index_col=False, names=['frame', 'ts1', 'ts2',
                                                                        'lin_vel_right', 'ang_vel_right'])
        data = pd.concat([data_left, data_right[['lin_vel_right', 'ang_vel_right']]], axis=1)
        self.timestamps = get_timestamps(data)
        self.wheel = np.column_stack((data['lin_vel_left'], data['ang_vel_left'],
                                        data['lin_vel_right'], data['ang_vel_right']))
    
    def get_frame(self, frame_id:int) -> tuple:
        """
        Returns frame timestamp, left wheel linear and angular velocities,\n
        right wheel linear and angular velocities.
        """
        return self.timestamps[frame_id], self.wheel[frame_id]
    
    def get_frame_by_timestamps(self, timestamp:int) -> tuple:
        """
        Returns frame timestamp, left wheel linear and angular velocities,\n
        right wheel linear and angular velocities.
        """
        frame = find_nearest(self.timestamps, timestamp)
        return self.get_frame(frame)

class CALIBRATION(object):
    
    def __init__(self, path:str, type:str):
        self.path = os.path.join(path, f'5_CALIBRATION_{type}')
    
    def get_camera_calib(self) -> tuple:
        """
        Returns Left and Right Camera Intrinsics(K), Radial Distortions(D),\n
        Fundamental(F) and Essential(E) Matrices.\n
        Also returns Left->Right Camera Rotation(R) and Translation(t).
        """
        contents = open(f'{self.path}/CALIBRATION_CAMERA.txt').read()
        K = re.findall(r'Intrinsic \(K\)(.*?)Radial Distortion', contents, re.DOTALL)
        K = [np.array(clean_matrices(matrix)) for matrix in K]
        D = re.findall(r'Radial Distortion\s*\(D\)(.*?)\n\n', contents, re.DOTALL)
        D = [np.squeeze(np.array(clean_matrices(matrix))) for matrix in D]
        F = re.search(r'3. Fundamental Matrix(.*?)4. Essential Matrix', contents, re.DOTALL)
        F = np.array(clean_matrices(F.group(1))) if F else None
        E = re.search(r'4. Essential Matrix(.*?)5. Rotation', contents, re.DOTALL)
        E = np.array(clean_matrices(E.group(1))) if E else None
        R = re.search(r'5. Rotation(.*?)6. Translation \(mm scale\)', contents, re.DOTALL)
        R = np.array(clean_matrices(R.group(1))) if R else None
        t = re.search(r'6. Translation \(mm scale\)(.*?)$', contents, re.DOTALL)
        t = np.squeeze(np.array(clean_matrices(t.group(1)))/1e3) if t else None
        return K, D, F, E, R, t
    
    def get_camera_lidar_calib(self) -> tuple:
        """
        Returns LIDAR to Camera(Left) Rotation(R) and Translation(t).
        """
        contents = open(f'{self.path}/CALIBRATION_CAMERA_LIDAR.txt').read()
        R = re.search(r'Rotation Matrix \(attitude of LiDAR in camera coordinate\)(.*?)\n\n', contents, re.DOTALL)
        R = np.array(clean_matrices(R.group(1))) if R else None
        t = re.search(r'Translation \(location of LiDAR in camera coordinate\)(.*?)\n\n', contents, re.DOTALL)
        t = np.squeeze(np.array(clean_matrices(t.group(1)))) if t else None
        return R, t
    
    def get_camera_radar_calib(self) -> tuple:
        """
        Returns RADAR to Camera(Left) Rotation(R) and Translation(t).
        """
        contents = open(f'{self.path}/CALIBRATION_CAMERA_RADAR.txt').read()
        R = re.search(r'Rotation Matrix \(attitude of RADAR in camera coordinate\)(.*?)\n\n', contents, re.DOTALL)
        R = np.array(clean_matrices(R.group(1))) if R else None
        t = re.search(r'Translation \(location of RADAR in camera coordinate\)(.*?)\n\n', contents, re.DOTALL)
        t = np.squeeze(np.array(clean_matrices(t.group(1)))) if t else None
        return R, t
    
    def get_camera_imu_calib(self) -> tuple:
        """
        Returns IMU to Camera (Left & Right) Transformation(T) Matrices.\n
        [IMU->Left, IMU->Right]
        """
        contents = re.sub('\[|\]', '', open(f'{self.path}/CALIBRATION_CAMERA_IMU.txt').read())
        T_ci = re.findall(r'T_ci:  \(imu0 to cam[01]\):(.*?)T_ic', contents, re.DOTALL)
        T_ci = [np.array(clean_matrices(matrix)) for matrix in T_ci]
        return T_ci
    
    ##### For more, refer to calibration files #####

class DeltaTime(object):
    """
    Class to measure time difference between frames.
    """

    def __init__(self):
        self.prev_timestamp = None
    
    def update(self, timestamp):
        """
        Measures time difference in seconds between timestamps.
        """
        timestamp_diff = timestamp - self.prev_timestamp if self.prev_timestamp is not None else 0
        self.prev_timestamp = timestamp
        return timestamp_diff * 1e-9