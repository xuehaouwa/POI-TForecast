import torch
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
from torch.utils.data import DataLoader
from gv_tools.util.logger import Logger
from dataset import GowallaSocialDataset
from torch.autograd import Variable
import os
from modules.net import M1


class Trainer:
    def __init__(self, cfg_params, args, logger: Logger, result_logger: Logger):
        self.cfg = cfg_params
        self.args = args
        cfg_params.copyAttrib(self)
        self.__dict__.update(args.__dict__)
        self.logger = logger
        self.res_logger = result_logger
        self.train_dataloader = None
        self.test_dataloader = None

        self.net = None
        self.loss_func = torch.nn.NLLLoss()
        self.loss_2 = torch.nn.MSELoss()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.loc = np.load(self.poi_path)[:, :2]
        scaler.fit(self.loc)
        self.normalized_loc = torch.from_numpy(scaler.transform(self.loc)).cuda()
        self.normalized_loc_np = scaler.transform(self.loc)

    def build_data_loader(self):
        self.logger.log('... Start Building Data Loaders ...')
        train_social = json.load(open(os.path.join(self.train_path, "social.json")))
        train_x = np.load(os.path.join(self.train_path, "input_seq.npy"))
        train_user = np.load(os.path.join(self.train_path, "input_user.npy"))
        train_y = np.load(os.path.join(self.train_path, "output.npy"))
        test_social = json.load(open(os.path.join(self.test_path, "social.json")))
        test_x = np.load(os.path.join(self.test_path, "input_seq.npy"))
        test_user = np.load(os.path.join(self.test_path, "input_user.npy"))
        test_y = np.load(os.path.join(self.test_path, "output.npy"))
        train_set = GowallaSocialDataset(self.cfg, train_x, train_user, train_y, train_social)
        test_set = GowallaSocialDataset(self.cfg, test_x, test_user, test_y, test_social)
        self.train_dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        self.logger.log('... Data Loader Done ...')

    def build_model(self, pretrained):
        self.logger.log('... Building model network ...')
        if self.model_type == "M1":
            self.net = M1(self.cfg, self.args)
        else:
            print("WRONG MODEL TYPE")
            exit()

        if self.use_cuda:
            self.net.cuda()
            self.logger.log('USE_GPU = True')

        if pretrained is not None:
            self.logger.log("Loading Pretrained Model")
            self.net.load_state_dict(torch.load(pretrained))
            self.test(epoch="pretrain")

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr)

        self.logger.log("... Network Build ...")
        self.logger.log(f"learning rate: {self.lr}")

    def train(self):
        self.logger.log("... Training Started ...")
        epoch = 0

        while epoch < self.epochs:
            epoch += 1
            losses = []
            loss_2 = []
            loss_3 = []
            self.net.train()

            for i, (data_x, data_user, data_y, input_semantic, input_tid, input_loc, out_loc, input_social, social_tid, social_semantic) in tqdm(enumerate(self.train_dataloader)):

                if self.use_cuda:
                    data_x, data_user, target, input_semantic, input_tid, input_loc, out_loc, input_social, social_tid, social_semantic = Variable(
                        data_x.to(torch.int64)).cuda(), Variable(
                        data_user.to(torch.int64)).cuda(), Variable(
                        data_y.to(torch.int64)).cuda(), Variable(
                        input_semantic.to(torch.int64)).cuda(), Variable(
                        input_tid.to(torch.int64)).cuda(), Variable(
                        input_loc).cuda(), Variable(
                        out_loc).cuda(), Variable(
                        input_social.to(torch.int64)).cuda(), Variable(social_tid.to(torch.int64)).cuda(), Variable(social_semantic.to(torch.int64)).cuda()
                else:
                    data_x, data_user, target, input_semantic, input_tid, input_loc, out_loc, input_social, social_tid, social_semantic = Variable(
                        data_x.to(torch.int64)), Variable(
                        data_user.to(torch.int64)), Variable(
                        data_y.to(torch.int64)), Variable(
                        input_semantic.to(torch.int64)), Variable(
                        input_tid.to(torch.int64)), Variable(
                        input_loc), Variable(
                        out_loc), Variable(
                        input_social.to(torch.int64)), Variable(social_tid.to(torch.int64)), Variable(social_semantic.to(torch.int64))

                self.optimizer.zero_grad()
                out, pred_loc = self.net(data_x, data_user, data_extra, input_time, pred_time, input_semantic, input_tid, pred_tid, input_loc, input_social, social_tid, social_semantic)
                if self.alpha > 0 and self.beta > 0:
                    rmse_loss = self.loss_2(pred_loc, out_loc)
                    consistant_loss = self.get_consist_loss(out, pred_loc)
                    loss = self.loss_func(out, target.squeeze(-1)) + self.alpha * rmse_loss + self.beta * consistant_loss
                elif self.alpha == -1 and self.beta == 0:
                    loss = self.loss_2(pred_loc, out_loc)
                else:
                    loss = self.loss_func(out, target.squeeze(-1))
                loss.backward()
                # update the weights
                self.optimizer.step()
                losses.append(loss.item())
                if self.alpha > 0 and self.beta > 0:
                    loss_2.append(rmse_loss.item())
                    loss_3.append(consistant_loss.item())

            if epoch % self.verbose_step == 0:
                self.logger.field('Epoch', epoch)
                self.logger.field('loss', np.mean(losses))
                if self.alpha > 0 and self.beta > 0:
                    self.logger.field('mse_loss', np.mean(loss_2))
                    self.logger.field('consistant_loss', np.mean(loss_3))
            
            
            # if epoch % self.interval_test_epochs == 0 and epoch < self.epochs:
                # reserved for val, update test() with val_loader

            if epoch == self.epochs:
                self.test(epoch)
                self.save_model(name="end")

    def test(self, epoch):
        self.net.eval()
        self.res_logger.field("Testing Epoch", epoch)
        count = 0
        acc = np.zeros(4)
        with torch.no_grad():
            for i, (data_x, data_user, data_y, input_semantic, input_tid, input_loc, out_loc, input_social, social_tid, social_semantic) in tqdm(enumerate(self.test_dataloader)):

                if self.use_cuda:
                    data_x, data_user, target, input_semantic, input_tid, input_loc, out_loc, input_social, social_tid, social_semantic = Variable(
                        data_x.to(torch.int64)).cuda(), Variable(
                        data_user.to(torch.int64)).cuda(), Variable(
                        data_y.to(torch.int64)).cuda(), Variable(
                        input_semantic.to(torch.int64)).cuda(), Variable(
                        input_tid.to(torch.int64)).cuda(), Variable(
                        input_loc).cuda(), Variable(
                        out_loc).cuda(), Variable(
                        input_social.to(torch.int64)).cuda(), Variable(social_tid.to(torch.int64)).cuda(), Variable(social_semantic.to(torch.int64)).cuda()
                else:
                    data_x, data_user, target, input_semantic, input_tid, input_loc, out_loc, input_social, social_tid, social_semantic = Variable(
                        data_x.to(torch.int64)), Variable(
                        data_user.to(torch.int64)), Variable(
                        data_y.to(torch.int64)), Variable(
                        input_semantic.to(torch.int64)), Variable(
                        input_tid.to(torch.int64)), Variable(
                        input_loc), Variable(
                        out_loc), Variable(
                        input_social.to(torch.int64)), Variable(social_tid.to(torch.int64)), Variable(social_semantic.to(torch.int64))

                out, _ = self.net(data_x, data_user, input_semantic, input_tid, input_loc, input_social, social_tid,
                                  social_semantic)

                for m in range(data_x.size(0)):
                    count += 1
                    acc_one, pred = self.get_acc(out[m], target[m].item())
                    acc = np.add(acc, acc_one)

        acc_1 = acc[0] / count

        self.res_logger.field("TOP-1_ACC", acc_1)
        self.res_logger.field("TOP-5_ACC", acc[1] / count)
        self.res_logger.field("TOP-10_ACC", acc[2] / count)
        self.res_logger.field("TOP-20_ACC", acc[3] / count)

        return acc_1

    def get_acc(self, net_output, target):
        if self.alpha == -1:
            _, topi = net_output.data.topk(20, largest=False)
        else:
            _, topi = net_output.data.topk(20)
        acc = np.zeros(4)
        index = topi.view(-1).cpu().numpy()
        if target == index[0]:
            acc[0] += 1
        if target in index[:5]:
            acc[1] += 1
        if target in index[:10]:
            acc[2] += 1
        if target in index[:20]:
            acc[3] += 1
        return acc, index[0]

    def get_consist_loss(self, net_output, target_loc):
        _, loc_id = torch.max(net_output, dim=1)
        loc = self.normalized_loc[loc_id.cpu() - 1]

        return self.loss_2(loc, target_loc)

    def save_model(self, name):
        '''
        save the current model in self.net to the specified file name
        and path name.
        '''
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.net.state_dict(),
                   os.path.join(self.save_path, name + '_params.pkl'))
        torch.save(self.net, os.path.join(self.save_path, name + '.pkl'))



