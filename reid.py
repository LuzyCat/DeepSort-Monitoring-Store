import numpy as np
from sklearn import cluster

import torch
from torchreid import metrics
from scipy.spatial.distance import cosine, cdist
from yolov5.utils.general import xywh2xyxy

class ClusterFeature:
    def __init__(self, dist_metric='cosine'):
        self.clusters = []
        self.clusters_sizes = []
        self.dist_metric = dist_metric
            
    def update(self, feature, threshold = 0.2, exist_id = -1):
        feature_vec = feature
        if len(self.clusters) == 0:
            self.__add_new_cluster(feature_vec)
            return 1
        else:
            distance = cdist(feature_vec.reshape(1, -1),
                        np.array(self.clusters).reshape(len(self.clusters), -1), self.dist_metric)
            nearest_idx = np.argmin(distance)
            
            if exist_id != -1:
                # if nearest_idx != exist_id-1 and distance[0, nearest_idx] < threshold:
                #     print("If you need to Change ID?")
                # return exist_id
                if nearest_idx == exist_id-1 or distance[0, nearest_idx] >= threshold:
                    self.clusters_sizes[exist_id-1] += 1
                    self.clusters[exist_id-1] += (feature_vec - self.clusters[exist_id-1]) / self.clusters_sizes[exist_id-1]
                return exist_id
            else:
                if distance[0, nearest_idx] < threshold:
                    # self.clusters_sizes[nearest_idx] += 1
                    # self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) / self.clusters_sizes[nearest_idx]
                    return nearest_idx + 1
                else:
                    self.__add_new_cluster(feature_vec)
                    return len(self.clusters)
    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)
    def get_cluster_by_id(self, id):
        return np.array(self.clusters[id])
    def get_n_clusters(self):
        return len(self.clusters)
    def __add_new_cluster(self, feature_vec):
        self.clusters.append(feature_vec)
        self.clusters_sizes.append(1)
    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        return distmat.numpy()

class MultiReID:
    def __init__(self, n_resources=1, occlusion_thresh = 0.7, dist_thresh = 0.2, init_global = 0):
        self.dist_metric = 'cosine'
        self.occlusion_threshold = occlusion_thresh
        self.dist_threshold = dist_thresh
        self.global_ids = []
        self.last_global_id = init_global
        self.n_resources = n_resources
        self.match_id = []
        self.feature_avg = []
        self.window = 10
        for x in range(0, n_resources):
            self.match_id.append(dict())
            self.feature_avg.append(dict())
        self.feature_clusters = ClusterFeature(self.dist_metric)
        
    def update(self, path, outputs, features, exist_ids, det_ROI):
        """
        update frame

        Args:
            path (str): camera numbers
            outputs (ndarray): inference result np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int)
            features (ndarray): extracted features
        """
        c_id = int(path)
        occlusion = self.__check_occlusion(outputs, det_ROI)
        # exist_in_frame = []
        
        fused_id = []
        
        for i, (output, feature) in enumerate(zip(outputs, features)):
            id = output[4]
            feature = feature.detach().cpu().numpy()
            fused_id.append(id * -1)
            if occlusion[i]:
                if self.match_id[c_id].get(id, -1) != -1 and self.match_id[c_id][id] not in fused_id:
                    fused_id[-1] = self.match_id[c_id][id]
            else:
                # if exist_ids.get(id, -1) == -1:
                #     self.feature_avg[c_id][id] = feature
                # elif exist_ids[id] < self.window:
                #     self.feature_avg[c_id][id] += (feature - self.feature_avg[c_id][id]) / (exist_ids[id] + 1)
                # else:
                if self.match_id[c_id].get(id, -1) == -1:
                    cluster_id = self.feature_clusters.update(feature, self.dist_threshold)
                    if len(self.global_ids) == 0:
                        if cluster_id != 1 :
                            print("ERROR: INVALID ID")
                            continue
                    if len(fused_id) > 0 and (cluster_id in fused_id):
                        continue
                    self.match_id[c_id][id] = cluster_id + self.last_global_id
                    self.global_ids.append(cluster_id)
                    fused_id[-1] = cluster_id
                else:
                    cluster_id = self.feature_clusters.update(feature, self.dist_threshold, exist_id=self.match_id[c_id][id])
                    # if cluster_id != self.match_id[c_id][id] and cluster_id not in fused_id:
                    #     self.match_id[c_id][id] = cluster_id
                    # if exist_ids[id] % 10 == 0:
                    #     cluster_id = self.feature_clusters.update(feature, self.dist_threshold)
                    #     if self.match_id[c_id][id] != cluster_id:
                    #         self.match_id[c_id][id] = cluster_id
                    #         self.global_ids.append(cluster_id)
                    # else:   
                    #     cluster_id = self.feature_clusters.update(feature, self.dist_threshold, exist_id=self.match_id[c_id][id])
                    # if exist_ids[id] % 10 == 0:
                    #     cluster_id = self.feature_clusters.update(feature, self.dist_threshold, exist_id=self.match_id[c_id][id])
                    fused_id[-1] = self.match_id[c_id][id]
        return fused_id
    
    def get_last_global_id(self):
        return self.feature_clusters.get_n_clusters()
    
    def __check_occlusion(self, detections, ROI):
        n_detections = detections.shape[0]
        if len(ROI) >= 2:
            ROI_box = [ROI[-2][0], ROI[-2][1], ROI[-1][0], ROI[-1][1]]
        # occlusion = np.zeros((1, n_detections), dtype=np.int8)
        occlusion = [False] * n_detections
        for i in range(0, n_detections):
            det1 = detections[i]
            if len(ROI) >= 2:
                if self.__ios(det1[0:4], ROI_box) > 0.1:
                    occlusion[i] = True
                    break
            for j in range(i+1, n_detections):
                det2 = detections[j]
                if self.__ios(det1[0:4], det2[0:4]) > self.occlusion_threshold:
                    occlusion[i] = True
                    break
        return occlusion
                
    def __ios(self, b1, b2, a1=None, a2=None):
        # intersection over self
        if a1 is None:
            a1 = self._area(b1)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])
        return intersection / a1 if a1 > 0 else 0
    def _area(self, box):
        return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)
    def _xywh2xxyy(self, box):
        result = box
        result[0] = box[0] - box[2] / 2  # top left x
        result[1] = box[1] - box[3] / 2  # top left y
        result[2] = box[0] + box[2] / 2  # bottom right x
        result[3] = box[1] + box[3] / 2  # bottom right y
        return result
 
        
        