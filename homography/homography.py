import numpy as np
import cv2

class HomographyProjector:
    def __init__(self, court_image_points, court_real_points,
                 net_image_points, net_real_points):
        # Гомография для поля (вид сверху)
        self.H_field, _ = cv2.findHomography(court_image_points, court_real_points)

        # Гомография для сетки (OxZ) — только X и Z
        self.H_net, _ = cv2.findHomography(net_image_points, net_real_points)

    def project_point(self, image_point, plane='net'):
        """Проецирует точку (x, y) с изображения в плоскость 'field' или 'net'."""
        point = np.array([[image_point]], dtype=np.float32)  # [1, 1, 2]
        if plane == 'field':
            projected = cv2.perspectiveTransform(point, self.H_field)
        elif plane == 'net':
            projected = cv2.perspectiveTransform(point, self.H_net)
        else:
            raise ValueError("Plane must be 'field' or 'net'")
        return projected[0][0].tolist()