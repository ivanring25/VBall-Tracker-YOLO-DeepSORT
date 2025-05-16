import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Primary team colors
        self.libero_colors = {}  # Libero colors per team
        self.player_team_dict = {}  # Player ID to team mapping
        self.kmeans = None
        self.color_threshold = 100  # Distance threshold for libero detection (adjustable)

    def get_clustering_model(self, image, n_clusters=2):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]

        h, w = player_img.shape[:2]
        if h < 5 or w < 5:
            return np.array([0, 0, 0])

        # Focus on the central region of the bbox
        center_x_start = int(w * 0.3)  # Start at 30% of width
        center_x_end = int(w * 0.7)    # End at 70% of width
        center_y_start = int(h * 0.3)  # Start at 30% of height
        center_y_end = int(h * 0.7)    # End at 70% of height

        # Extract the central patch
        center_patch = player_img[center_y_start:center_y_end, center_x_start:center_x_end]

        # Ensure the central patch is valid
        if center_patch.size == 0 or center_patch.shape[0] < 5 or center_patch.shape[1] < 5:
            return np.array([0, 0, 0])

        # Apply KMeans clustering to the central patch
        kmeans = self.get_clustering_model(center_patch, n_clusters=2)
        labels = kmeans.labels_.reshape(center_patch.shape[0], center_patch.shape[1])

        # Assume the dominant cluster in the central region is the uniform color
        unique, counts = np.unique(labels, return_counts=True)
        uniform_cluster = unique[np.argmax(counts)]  # Most frequent cluster

        return kmeans.cluster_centers_[uniform_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        player_ids = []
        for player_id, data in player_detections.items():
            bbox = data["bbox"]
            color = self.get_player_color(frame, bbox)
            player_colors.append(color)
            player_ids.append(player_id)

        player_colors = np.array(player_colors)
        # Use more clusters to potentially separate liberos
        kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
        labels = kmeans.fit_predict(player_colors)

        # Identify the two largest clusters as primary team colors
        unique, counts = np.unique(labels, return_counts=True)
        top_clusters = unique[np.argsort(counts)[-2:]]  # Get the two most frequent clusters
        self.team_colors[1] = kmeans.cluster_centers_[top_clusters[0]]
        self.team_colors[2] = kmeans.cluster_centers_[top_clusters[1]]

        # Identify potential libero colors (smaller clusters)
        for cluster in unique:
            if cluster not in top_clusters:
                # Assign libero colors to the closest team
                distances = [
                    np.linalg.norm(kmeans.cluster_centers_[cluster] - self.team_colors[team_id])
                    for team_id in [1, 2]
                ]
                team_id = 1 if distances[0] < distances[1] else 2
                self.libero_colors.setdefault(team_id, []).append(kmeans.cluster_centers_[cluster])

        self.kmeans = kmeans

        # Assign players to teams based on initial clustering
        for player_id, label in zip(player_ids, labels):
            if label in top_clusters:
                team_id = 1 if label == top_clusters[0] else 2
            else:
                # For libero-like colors, assign to the closest team
                color = player_colors[player_ids.index(player_id)]
                distances = [np.linalg.norm(color - self.team_colors[team_id]) for team_id in [1, 2]]
                team_id = 1 if distances[0] < distances[1] else 2
            self.player_team_dict[player_id] = team_id

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            self.player_team_dict[player_id] = None
            return None

        player_color = self.get_player_color(frame, player_bbox)
        # Check if the color is closer to a libero color
        min_distance = float('inf')
        assigned_team = None

        for team_id in [1, 2]:
            # Check distance to primary team color
            team_distance = np.linalg.norm(player_color - self.team_colors[team_id])
            if team_distance < min_distance:
                min_distance = team_distance
                assigned_team = team_id

            # Check distance to libero colors for this team
            if team_id in self.libero_colors:
                for libero_color in self.libero_colors[team_id]:
                    libero_distance = np.linalg.norm(player_color - libero_color)
                    if libero_distance < min_distance and libero_distance < self.color_threshold:
                        min_distance = libero_distance
                        assigned_team = team_id

        self.player_team_dict[player_id] = assigned_team
        return assigned_team