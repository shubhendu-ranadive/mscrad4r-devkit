import argparse
from data_utils import *
from vis_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Path to data directory.')
    parser.add_argument('-v', '--vis', action='store_true', help='Visualize result')
    args = parser.parse_args()
    return args

class Args(object):
    data_dir = '../Downloads/Datasets/MSC-Rad4R/URBAN/URBAN_A0'
    vis = False

def main():

    # args = parse_args()
    args = Args()

    # imu = IMU(args.data_dir)
    # gps = GPS(args.data_dir)
    lidar = LIDAR(args.data_dir)
    radar = RADAR(args.data_dir)
    image = IMAGE(args.data_dir)
    delta_time = DeltaTime()
    calib = CALIBRATION(args.data_dir, 'URBAN')

    K, D, F, E, _, _ = calib.get_camera_calib()
    # R, t = calib.get_camera_lidar_calib()
    R, t = calib.get_camera_radar_calib()
    # T = calib.get_camera_imu_calib()
    left_img = DrawImage('left image', K[0])
    left_img.set_rot_trans(R, t)

    for i, ts in enumerate(lidar.timestamps):
        # lidar_ts, lidar_pts = lidar.get_frame(i)
        radar_ts, radar_pts  = radar.get_frame_by_timestamp(ts)
        img_ts, img_left, img_right = image.get_frame_by_timestamp(ts)

        # TO DO: Add colors
        # colors = (np.ones((lidar_pts.shape[0], 3)) * 127).tolist()
        colors = (np.ones((radar_pts.shape[0], 3)) * 127).tolist()

        # img = left_img.draw_points_on_img(img_left, lidar_pts[:, :3], colors)
        img = left_img.draw_points_on_img(img_left, radar_pts[:, :3], colors)
        # wheel_ts, wheel_data = wheel.get_frame_by_timestamps(ts)
        # imu_ts, imu_vel, imu_acc, _ = imu.get_frame_by_timestamps(ts)
        # delta = delta_time.update(ts)
        cv2.imshow('Lidar Projection', img)

        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()