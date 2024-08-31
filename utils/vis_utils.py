import cv2
import numpy as np

class DrawImage(object):
    """
    Class to represent sensor data on Image.\n
    Mainly for LiDAR and RADAR data.
    """
    def __init__(self, name:str, K:np.ndarray) -> None:
        self.name = name
        self.intrinsic = K
        self.rot = None
        self.trans = None
    
    def set_rot_trans(self, R:np.ndarray, t:np.ndarray) -> None:
        """
        Set Rotation and Translation matrices for Transformation
        """
        self.rot = R
        self.trans = t

    def sensor2img(self) -> np.ndarray:
        """
        Calculate Sensor->Image Transformation Matrix.\n
        Takes Senor->Camera Rotation, Translation and\n 
        Camera Intrinsic Matrices as arguments.
        """
        assert self.rot.size == 9 and self.trans.size == 3, \
            "Set correct 3x3 Rotation Matrix and 1x3 Translation vector!!"
        if self.intrinsic.size == 16:
            K_hg = self.intrinsic
        else:
            K_hg = np.eye(4)
            K_hg[:3, :3] = self.intrinsic
        
        T = np.eye(4)
        T[:3, :3] = self.rot
        T[:3, 3] = self.trans

        return K_hg @ T

    def draw_points_on_img(self, img:np.ndarray, pts:np.ndarray, color) -> np.ndarray:
        """
        Returns image with points drawn on it.
        """
        img = img.copy()
        s2i_rt = self.sensor2img()
        pts = pts.copy()
        N = pts.shape[0]
        pts_4d = np.concatenate([pts.reshape(-1, 3), np.ones((N, 1))], axis=-1)
        pts_4d = (s2i_rt @ pts_4d.T).T
        pts_4d[:, 2] = np.clip(pts_4d[:, 2], a_min=1e-5, a_max=1e5)
        pts_4d[:, 0] /= pts_4d[:, 2]
        pts_4d[:, 1] /= pts_4d[:, 2]
        pts_2d = pts_4d[:, :2].reshape(N, 2)
        pts_2d = np.clip(pts_2d, -1e4, 1e5).astype(np.int32)

        for i, pt in enumerate(pts_2d):
            cv2.circle(img, pt.tolist(), 2, color[i], -1)
        return img

class BEV(object):
    """
    Class to represent objects in Bird-Eye-View(BEV).
    """

    def __init__(self, name:str, x_range=(-15, 15), y_range=(-15, 15), step=60) -> None:
        self.name = name
        self.x_range = x_range
        self.y_range = y_range
        self.step = step
        self.bev_h = step * (x_range[1] - x_range[0])
        self.bev_w = step * (y_range[1] - y_range[0])
        self.bev = np.zeros((self.bev_h, self.bev_w, 3), np.uint8)
    
    def clear_bev(self) -> None:
        """
        Reset BEV.
        """
        self.bev[:] = 0
    
    def get_v(self, x:float) -> int:
        """
        Get BEV image x pixel coordinate.
        """
        return self.bev_h - 1 - round(self.step * (x - self.x_range[0]))
    
    def get_u(self, y:float) -> int:
        """
        Get BEV image y pixel coordinate.
        """
        return round(self.step * (-y - self.y_range[0]))
    
    def get_point(self, x:float, y:float) -> tuple:
        """
        Return BEV image x,y pixel coordinates
        """
        return (self.get_u(y), self.get_v(x))
    
    def draw_grid(self, step=1) -> None:
        """
        Draw grid on BEV.
        """
        color = (127, 127, 127)
        for x in range(self.x_range[0], self.x_range[1], step):
            v = self.get_v(x)
            cv2.line(self.bev, (0, v), (self.bev_w, v), color)
        for y in range(self.y_range[0], self.y_range[1], step):
            u = self.get_u(y)
            cv2.line(self.bev, (u, 0), (u, self.bev_h), color)
    
    def draw_points(self, xs:tuple, ys:tuple, colors:tuple) -> None:
        """
        Draw points on BEV.
        """
        for x, y, c in zip(xs ,ys, colors):
            pt = self.get_point(x, y)
            cv2.circle(self.bev, pt, 1, c, 2)
    
    def show(self) -> None:
        """
        Show BEV on output window.
        """
        cv2.imshow(self.name, self.bev)
    
    def save(self, dst:str) -> None:
        """
        Save BEV image to a destination.
        """
        cv2.imwrite(dst, self.bev)
    
    def get_bev(self) -> np.ndarray:
        """
        Get BEV image. 
        """
        return self.bev.copy()