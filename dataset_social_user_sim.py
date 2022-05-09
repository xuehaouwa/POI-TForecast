from tqdm import tqdm
import numpy as np
import json
import os
import datetime


class DataLoad:
    def __init__(self, cfg_params):
        cfg_params.copyAttrib(self)
        self.raw_data = np.load(self.data_path)
        self.social = json.load(open(self.social_path))
        self.user_sequence = {}
        self.load()

    def load(self):
        for i in range(self.user_num):
            u_id = i + 1
            self.user_sequence[u_id] = self.get_user(u_id)

    def get_user(self, u_id):
        rows = (self.raw_data[:, 0] == u_id)
        selected = self.raw_data[rows, :]

        out = []
        for d in selected:
            out.append([int(d[-1]), int(d[1])])

        sorted_out = sorted(out, key=lambda x: x[1])
        return sorted_out

    def get_extra(self, tid):
        extra_time = np.zeros((1, self.interval))
        extra_day = np.zeros((1, self.weekday))
        actual_time = datetime.datetime(2012, 1, 1) + datetime.timedelta(seconds=tid*3600)
        extra_day[0][actual_time.weekday()] = 1
        extra_time[0][actual_time.hour] = 1

        return np.concatenate((extra_day, extra_time), axis=-1)

    def find_neighbour(self, user_data, sequence_data):
        social_seq_dict = {}

        for i in tqdm(range(len(user_data))):
            user_i = user_data[i][0]
            social_seq_dict[i] = []
            if str(user_i) in self.social.keys():
                last_time_i = sequence_data[i][-1][-1]
                if len(self.social[str(user_i)]) > 0:
                    for s in range(len(sequence_data)):
                        user_j = user_data[s][0]
                        if user_j in self.social[str(user_i)]:
                            if sequence_data[s][-1][-1] <= last_time_i:
                                social_seq_dict[i].append(s)
            else:
                continue

        return social_seq_dict

    def get_train(self):
        input_sequence = []
        input_user = []
        output = []
        extra = []
        for u_id in self.user_sequence.keys():
            num_sample = int(len(self.user_sequence[u_id]) * self.train_split)
            sequence = self.user_sequence[u_id][0: num_sample]
            num = len(sequence) - self.obs_len
            if num > 0:
                for i in range(num):
                    input_user.append(u_id)
                    input_sequence.append(sequence[i: i+self.obs_len])
                    output.append(sequence[i + self.obs_len])
                    extra.append(self.get_extra(sequence[i + self.obs_len][-1]))
            else:
                print(f"Not long enough train sequence user {u_id}")

        input_sequence = np.reshape(input_sequence, [-1, self.obs_len, 2])
        input_user = np.reshape(input_user, [-1, 1])
        output = np.reshape(output, [-1, 2])
        extra = np.reshape(extra, [-1, self.weekday+self.interval])
        train_social_dict = self.find_neighbour(input_user, input_sequence)
        print(f"Total number of training samples: {len(input_sequence)}")
        print(f"input_sequence shape {np.shape(input_sequence)}")
        print(f"input_user shape {np.shape(input_user)}")

        return input_sequence, input_user, output, extra, train_social_dict

    def get_test(self):
        input_sequence = []
        input_user = []
        output = []
        extra = []
        for u_id in self.user_sequence.keys():
            num_sample = int(len(self.user_sequence[u_id]) * self.train_split)
            sequence = self.user_sequence[u_id][num_sample: ]
            num = len(sequence) - self.obs_len
            if num > 0:
                for i in range(num):
                    input_user.append(u_id)
                    input_sequence.append(sequence[i: i + self.obs_len])
                    output.append(sequence[i + self.obs_len])
                    extra.append(self.get_extra(sequence[i + self.obs_len][-1]))
            else:
                print(f"Not long enough train sequence user {u_id}")
                print(f"total_length: {len(self.user_sequence[u_id])}, test_len: {len(sequence)}")

        input_sequence = np.reshape(input_sequence, [-1, self.obs_len, 2])
        input_user = np.reshape(input_user, [-1, 1])
        output = np.reshape(output, [-1, 2])
        extra = np.reshape(extra, [-1, self.weekday + self.interval])
        test_social_dict = self.find_neighbour(input_user, input_sequence)
        print(f"Total number of testing samples: {len(input_sequence)}")
        print(f"input_sequence shape {np.shape(input_sequence)}")
        print(f"input_user shape {np.shape(input_user)}")

        return input_sequence, input_user, output, extra, test_social_dict


def save(input_sequence, input_user, output, extra, social_dict, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, "input_seq.npy"), input_sequence)
    np.save(os.path.join(save_path, "input_user.npy"), input_user)
    np.save(os.path.join(save_path, "output.npy"), output)
    np.save(os.path.join(save_path, "extra.npy"), extra)
    with open(os.path.join(save_path, "social.json"), "w") as f:
        json.dump(social_dict, f, indent=1)

