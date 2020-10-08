# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
import time
import cv2


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.kalman_trajectory = []
        self.kalman_predict = []
        self.center_pos_trajectory = []
        self.idx = 0

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        first_time = time.time()
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        self.whcrop = (x_crop.shape[2], x_crop.shape[3])
        outputs = self.model.track(x_crop)
        print("Time = ",time.time()-first_time)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr

            self.center_pos_trajectory.append(np.array([cx, cy]))
            self.idx = 0
            if len(self.center_pos_trajectory) > 2:
                self.calculate_Kalman(0)
        else:
            if len(self.center_pos_trajectory) > 2:
                self.calculate_Kalman(self.idx + 1)
                cx = self.kalman_predict[self.idx][0]
                cy = self.kalman_predict[self.idx][1] 
                if self.idx < 5:
                    self.idx += 1
            else:
                cx = self.center_pos[0]
                cy = self.center_pos[1] 

            width = self.size[0]
            height = self.size[1]

        # cx = bbox[0] + self.center_pos[0]
        # cy = bbox[1] + self.center_pos[1]

        # # smooth bbox
        # width = self.size[0] * (1 - lr) + bbox[2] * lr
        # height = self.size[1] * (1 - lr) + bbox[3] * lr

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])


        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
                'bbox': bbox,
                'best_score': best_score
               }

    def calculate_Kalman(self, numOfPredict):
        if len(self.center_pos_trajectory) > 2:
            kalman = Kalman(self.center_pos_trajectory[0])
            self.kalman_predict.clear()
            self.kalman_trajectory.clear()

            for i in range(1, len(self.center_pos_trajectory)):
                pos = kalman.kalmanPredict()
                self.kalman_trajectory.append(pos)
                kalman.kalmanCorrect(self.center_pos_trajectory[i])
            for i in range(numOfPredict):
                pos = kalman.kalmanPredict()
                self.kalman_predict.append(pos)
                kalman.kalmanCorrect(pos)

class Kalman:
    def __init__(self, point):
        self.measurement = np.zeros((2, 1), dtype=np.float32)
        self.measurement[0][0] = point[0]
        self.measurement[1][0] = point[1]
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePre[0, 0] = point[0]
        self.kalman.statePre[1, 0] = point[1]
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost[0, 0] = point[0]
        self.kalman.statePost[1, 0] = point[1]
        self.kalman.measurementMatrix = cv2.setIdentity(
            self.kalman.measurementMatrix)
        self.kalman.processNoiseCov = cv2.setIdentity(
            self.kalman.processNoiseCov, .01)
        self.kalman.measurementNoiseCov = cv2.setIdentity(
            self.kalman.measurementNoiseCov, .1)
        self.kalman.errorCovPost = cv2.setIdentity(
            self.kalman.errorCovPost, .1)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)

    def kalmanPredict(self):
        prediction = self.kalman.predict()
        predictPr = (prediction[0, 0], prediction[1, 0])
        return predictPr

    def kalmanCorrect(self, point):
        self.measurement[0, 0] = point[0]
        self.measurement[1, 0] = point[1]
        estimated = self.kalman.correct(self.measurement)
        return (estimated[0, 0], estimated[1, 0])