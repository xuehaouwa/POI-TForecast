import torch.utils.data as data
import torch
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime


def get_time(tid):
    actual_time = datetime.datetime(2009, 1, 1) + datetime.timedelta(seconds=int(tid) * 3600)
    day_index = actual_time.weekday()
    hour_index = actual_time.hour
    time_id = day_index * 24 + hour_index + 1

    return time_id


class GowallaSocialDataset(data.Dataset):
    def __init__(self, cfg_params, input_sequence, input_user, output, social):
        cfg_params.copyAttrib(self)
        self.social = social
        social_len = []
        for k, v in social.items():
            if len(v) > 0:
                social_len.append(len(v))
        self.max_neighbour = 30
        num_seq, obs_len, _ = np.shape(input_sequence)
        self.neighbour = np.zeros((num_seq, self.max_neighbour, obs_len))
        self.neighbour_tid = np.zeros((num_seq, self.max_neighbour, obs_len))
        self.neighbour_category = np.zeros((num_seq, self.max_neighbour, obs_len))
        tid_sequence = self.process_time(input_sequence[:, :, 1])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.loc = np.load(self.poi_path)[:, :2]
        scaler.fit(self.loc)
        self.normalized_loc = scaler.transform(self.loc)
        self.loc_x = self.normalized_loc[:, 0]
        self.loc_y = self.normalized_loc[:, 1]
        num_seq, obs_len, _ = np.shape(input_sequence)
        self.input_category = np.zeros((num_seq, obs_len))
        self.get_semantic(input_sequence)
        self.get_neighbours(input_sequence, tid_sequence)
        self.input_sequence = torch.Tensor(input_sequence[:, :, 0])
        self.input_time = torch.Tensor(input_sequence[:, :, 1])
        self.input_tid = torch.Tensor(tid_sequence)
        self.input_user = torch.Tensor(input_user)
        self.output = torch.Tensor(output[:, 0])
        self.input_category = torch.Tensor(self.input_category)
        self.input_loc = torch.Tensor(self.process_loc(input_sequence[:, :, 0]))
        self.output_loc = torch.Tensor(np.reshape(self.process_loc(output[:, 0]), (-1, 2)))

    def get_semantic(self, input_seq):
        poi = np.load(self.poi_path)[:, -1]
        for i in tqdm(range(len(self.input_category))):
            for j in range(self.obs_len):
                self.input_category[i, j] = int(poi[int(input_seq[i, j, 0]) - 1] + 2)

    def get_neighbours(self, input_seq, tid_seq):
        for i in tqdm(range(len(self.neighbour))):
            neighbour = self.social[str(i)]
            if len(neighbour) > 0:
                for n in range(len(neighbour)):
                    if n < self.max_neighbour:
                        self.neighbour[i, n, :] = input_seq[neighbour[n]][:, 0]
                        self.neighbour_tid[i, n, :] = tid_seq[neighbour[n]]
                        self.neighbour_category[i, n, :] = self.input_category[neighbour[n]]

    def process_time(self, raw_time):
        vfunc = np.vectorize(get_time)

        return vfunc(raw_time)

    def getx(self, poi):
        return self.loc_x[int(poi - 1)]

    def gety(self, poi):
        return self.loc_y[int(poi - 1)]

    def process_loc(self, poi):
        vx = np.vectorize(self.getx)
        vy = np.vectorize(self.gety)

        return np.concatenate((vx(poi), vy(poi)), axis=-1)

    def __len__(self):

        return len(self.input_user)

    def __getitem__(self, item):

        return self.input_sequence[item], self.input_user[item], self.output[item], self.input_category[item], \
               self.input_tid[item], self.input_loc[item], self.output_loc[item], self.neighbour[item], \
               self.neighbour_tid[item], self.neighbour_category[item]







