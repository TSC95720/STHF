import os
import torch
import torch.nn as nn
from bisect import bisect_right
from network import Model
from tools import os_walk, TripletLoss_WRT,SupConLoss

class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.pid_num

        self.module = config.module

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.c_learning_rate = config.c_learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones
        self.steps = config.steps

        self.img_h = config.img_h
        self.img_w = config.img_w
        self.weight_a = config.weight_a

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):

        self.model = Model(self.pid_num)
        self.model = self.model.to(self.device)

    def _init_creiteron(self):
        self.pid_creiteron = nn.CrossEntropyLoss()
        self.tri_creiteron = TripletLoss_WRT()
        self.con_creiteron = SupConLoss(self.device)

        self.pid_creiteron.to(self.device)
        self.tri_creiteron.to(self.device)

    def _init_optimizer(self):
        params = []
        keys = []
        for key, value in self.model.named_parameters():
            lr = self.learning_rate
            if 'classifier' in key:
                lr = self.learning_rate
            if 'classifier_i' in key:
                lr = self.learning_rate
            params += [{'params': [value], 'lr': lr, 'weight_decay': self.weight_decay}]
            keys += [[key]]

        self.model_optimizer = getattr(torch.optim, 'Adam')(params)
        self.model_lr_scheduler = WarmupMultiStepLR(self.model_optimizer, self.milestones,
                                                           gamma=0.1, warmup_factor=0.01, warmup_iters=10)  # 从warmup_factor*lr开始增长至lr(增长iters轮) 在miletones时衰减gamma*lr

    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'best_model.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pth' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        # model_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(resume_epoch))
        model_path = os.path.join(self.save_model_path,  'best_model.pth')
        self.model.load_state_dict(torch.load(model_path), strict=False)
        # print('Successfully resume shared_model from {}'.format(model_path)
        print('Successfully resume shared_model from best_model.pth')

    def set_train(self):
        self.model = self.model.train()
        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.training = False


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
