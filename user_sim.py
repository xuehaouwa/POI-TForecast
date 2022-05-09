from tqdm import tqdm
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from cfg.option import Options


class UserSim:
    def __init__(self, cfg_params):
        cfg_params.copyAttrib(self)
        self.raw_data = np.load(self.data_path)
        self.freq_mat = np.zeros((self.user_num, self.poi_num))
        self.count()
        self.sim = self.cosine_sim()

    def count(self):
        for i in tqdm(range(len(self.raw_data))):
            u_id = self.raw_data[i][0]
            p_id = self.raw_data[i][-1]
            self.freq_mat[int(u_id) - 1, int(p_id) - 1] += 1

    def cosine_sim(self):
        sim = cosine_similarity(self.freq_mat)
        return sim

    def save(self, thr, save_path="social.json"):
        a = np.argwhere(self.sim > thr)
        social = {}
        for i in range(len(a)):
            user_i = a[i][0] + 1
            user_j = a[i][1] + 1
            if user_i != user_j:
                if user_i not in social.keys():
                    social[int(user_i)] = [int(user_j)]
                else:
                    social[int(user_i)].append(int(user_j))

        with open(save_path, "w") as f:
            json.dump(social, f)

