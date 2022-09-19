from turtle import distance
import numpy as np
import torch
from torchreid import metrics
from scipy.spatial.distance import cdist

class FeatureMatrix:
    def __init__(self, dist_thresh=0.2, dist_metric='cosine', max = 50):
        self.n_saved_features = 0
        self.init_global_id = 1
        self.max = max
        self.distance_threshold = dist_thresh
        self.dist_metric = dist_metric
        self.feature_matrix = None
        self.feature_size = []
        
    def match_features(self, features):
        distance_matrix = self.compute_distance(features, self.feature_matrix)
        return distance_matrix
     
    def set_initial_matrix(self, features):
        self.feature_matrix = features
        self.n_saved_features = features.shape[0]
        self.feature_size = [1 for x in range(self.n_saved_features)]
        
    def add_feature(self, feature):
        self.feature_matrix = torch.cat((self.feature_matrix, feature.reshape(1, -1)), dim=0)
        self.n_saved_features += 1
        self.feature_size.append(1)
        if self.n_saved_features > self.max and self.feature_matrix.shape[0] > self.max:
            self.feature_matrix = self.feature_matrix[1:]
            self.feature_size = self.feature_size[1:]
            self.init_global_id += 1
            
    def update_feature(self, feature, col_num):
        self.feature_size[col_num] += 1
        self.feature_matrix[col_num] += (feature - self.feature_matrix[col_num]) / self.feature_size[col_num]
        
    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        return distmat.detach().cpu().numpy()
    
class MultiReID:
    def __init__(self, n_resources=1, occlusion_thresh=0.5, dist_thresh=0.2, recent_max = 50):
        self.dist_metric = 'cosine'
        self.occlusion_threshold = occlusion_thresh
        self.distance_threshold = dist_thresh
        self.n_resources = n_resources
        self.match_id = []
        self.feature_avg = []
        for x in range(n_resources):
            self.match_id.append(dict())
            self.feature_avg.append(dict())
        self.feature_matrix = FeatureMatrix(dist_thresh = self.distance_threshold, dist_metric = self.dist_metric, max=recent_max)
    def update(self, path, outputs, features, non_detecion_box, person_confs):
        """
        update frame

        Args:
            path (str): camera numbers
            outputs (ndarray): inference result np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int)
            features (ndarray): extracted features
        """
        cam_id = int(path)
        occlusion_matrix = self.__check_occlusion(outputs, non_detecion_box)
        n_detections = outputs.shape[0]
        fused_id = [-1 for x in range(n_detections)]
        
        if self.feature_matrix.n_saved_features == 0:
            self.feature_matrix.set_initial_matrix(features)
        else:
            distance_matrix = self.feature_matrix.match_features(features)
            # print(cam_id, distance_matrix)
            
            min_cols = np.argmin(distance_matrix, axis=1)
            min_cols = list(min_cols)
            unique_cols = set(min_cols)
            
            for i, (output, feature) in enumerate(zip(outputs, features)):
                local_id = output[4]
                if self.match_id[cam_id].get(local_id, -1) != -1:
                    fused_id[i] = self.match_id[cam_id][local_id]
                    if self.feature_matrix.init_global_id + min_cols[i] == fused_id[i] and person_confs[i] > 0.8 and \
                        distance_matrix[i, min_cols[i]] < self.distance_threshold and occlusion_matrix[i] == False:
                        self.feature_matrix.update_feature(feature, min_cols[i])
                    distance_matrix[:, min_cols[i]] = 1.0
                elif occlusion_matrix[i]:
                    fused_id[i] = local_id * -1
                else:
                    if min(distance_matrix[i]) > self.distance_threshold: # not matched
                        self.feature_matrix.add_feature(feature)
                        fused_id[i] = self.feature_matrix.n_saved_features
                        self.match_id[cam_id][local_id] = fused_id[i]
                    else:
                        if min_cols[i] in unique_cols:
                            if self.feature_matrix.init_global_id + min_cols[i] in fused_id:
                                fused_id[i] = local_id * -1
                            else:
                                if person_confs[i] > 0.8:
                                    self.feature_matrix.update_feature(feature, min_cols[i])
                                fused_id[i] = self.feature_matrix.init_global_id + min_cols[i]
                                self.match_id[cam_id][local_id] = fused_id[i]
                        else:
                            fused_id[i] = local_id * -1
                    unique_id = set(fused_id)
                    if fused_id[i] not in unique_id:
                        fused_id[i] = local_id * -1
        return fused_id
        
    def __check_occlusion(self, detections, non_detecion_box):
        n_detections = detections.shape[0] # number of detections in this frame
        if len(non_detecion_box) >= 2:
            ROI_box = [non_detecion_box[-2][0], non_detecion_box[-2][1], non_detecion_box[-1][0], non_detecion_box[-1][1]]
        occlusion_mat = [False] * n_detections
        for i in range(n_detections):
            det1 = detections[i]
            if len(non_detecion_box) >= 2:
                if ios(det1[0:4], ROI_box) > 0.1:
                    occlusion_mat[i] = True
                    break
            for j in range(i):
                det2 = detections[j]
                if ios(det1[0:4], det2[0:4]) > self.occlusion_threshold:
                    occlusion_mat[j] =True
                    break
        return occlusion_mat
        
                    
def ios(b1, b2, a1=None, a2=None):
    # intersection over self
    if a1 is None:
        a1 = area(b1)
    intersection = area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                min(b1[2], b2[2]), min(b1[3], b2[3])])
    return intersection / a1 if a1 > 0 else 0
def area(box):
        return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)
def xywh2xxyy(box):
    result = box
    result[0] = box[0] - box[2] / 2  # top left x
    result[1] = box[1] - box[3] / 2  # top left y
    result[2] = box[0] + box[2] / 2  # bottom right x
    result[3] = box[1] + box[3] / 2  # bottom right y
    return result
    