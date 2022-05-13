# -*- coding:utf-8 -*-
# @Time : 2020/12/30 17:16
# @Author: LCHJ
# @File : config.py
import numpy as np


class DefaultConfigs(object):
    def __init__(
        self,
        data_name=None,
        is_T=False,
        is_img_T=False,
        is_rgb=False,
        is_negative=False,
    ):
        # ========================  General Phd config =====================================
        # 1.data path
        self.data_name = data_name  # name of the datdset
        self.train_path = "./datasets/" + str(data_name)
        self.test_path = "./datasets/" + str(data_name)
        self.is_mat_dat = True  # Is the type of dataset .mat?
        self.is_T = is_T  # Do keywords of .mat need to be converted?
        if self.is_mat_dat:
            self.train_path = self.train_path + ".mat"
            self.test_path = self.test_path + ".mat"

        # 2.[num , features]
        self.is_img_T = is_img_T  # Does the image display need to rotate?
        self.is_rgb = is_rgb  # Is the image RGB  ?
        self.is_negative = is_negative

        self.input_shape = [28, 28, 1]  # image shape
        self.num_classes = 10  # class num
        # 3. Init loss and acc array
        self.loss_list = []
        self.best_acc = np.zeros(3)
        self.best_nmi = np.zeros(3)
        self.acc_init = 0
        self.nmi_init = 0
        self.acc_list = np.ones([1, 3], dtype=float) * 10
        self.nmi_list = np.ones([1, 3], dtype=float) * 10
        # 4. global parameter
        self.eval = False
        self.great = False

        # 7 learning rate
        self.D_lr = 1e-3
        self.D_weight_decay = 1e-5
        self.D_step_size = 4
        self.D_gamma = 0.95
        # ========================   Specific config for Specific data set =====================================
        if self.data_name == "umist_1024":
            self.train_path = "./datasets/umist_1024.mat"
            self.test_path = "./datasets/umist_1024.mat"

            self.is_img_T = True
            self.is_negative = True
            self.input_shape = [32, 32, 1]  # image shape
            self.num_classes = 20
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 401
            self.iteration_rate = (
                20  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 1000  # 575
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.hyper_parameter = [1, 0.01]
            # 7 learning rate
            self.D_lr = 1e-3
            self.D_weight_decay = 1e-5
            self.D_step_size = 4
            self.D_gamma = 0.95

        if self.data_name == "COIL20":
            self.train_path = "./datasets/COIL20.mat"
            self.test_path = "./datasets/COIL20.mat"

            self.is_img_T = True
            self.is_negative = True
            self.input_shape = [32, 32, 1]  # image shape
            self.num_classes = 20
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 401
            self.iteration_rate = (
                10  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 1000  # 1440
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.hyper_parameter = [1, 0.01]
            # 7 learning rate
            self.D_lr = 1e-3
            self.D_weight_decay = 1e-5
            self.D_step_size = 3
            self.D_gamma = 0.95

        if self.data_name == "MSRA":
            self.train_path = "./datasets/MSRA.mat"
            self.test_path = "./datasets/MSRA.mat"

            self.is_img_T = True
            self.input_shape = [16, 16, 1]  # image shape
            self.num_classes = 12
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 601
            self.iteration_rate = (
                20  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 600
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.Multiple_AHML = 1  # AHML
            self.Mul_MSN = 0.01  # MSE

        if self.data_name == "Palm":
            self.train_path = "./datasets/Palm.mat"
            self.test_path = "./datasets/Palm.mat"

            self.input_shape = [16, 16, 1]  # image shape
            self.num_classes = 100
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 128
            self.iteration_rate = (
                20  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 200
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.hyper_parameter = [1, 0.01]
            # 7 learning rate
            self.D_lr = 4e-4
            self.D_weight_decay = 1e-4
            self.D_gamma = 0.992

        if self.data_name == "USPS":
            self.train_path = "./datasets/USPS.mat"
            self.test_path = "./datasets/USPS.mat"

            self.input_shape = [16, 16, 1]  # image shape
            self.num_classes = 10
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 501
            self.iteration_rate = (
                5  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 1000
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.hyper_parameter = [1, 0.008]
            # 7 learning rate
            self.D_lr = 1e-3
            self.D_weight_decay = 1e-5
            self.D_step_size = 3
            self.D_gamma = 0.95

        if self.data_name == "fashion-mnist":
            self.train_path = "./datasets/fashion-mnist.mat"
            self.test_path = "./datasets/fashion-mnist.mat"
            # 2.[num , features]
            self.is_T = True
            self.is_img_T = True
            self.input_shape = [28, 28, 1]  # image shape
            self.num_classes = 10
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 501
            self.iteration_rate = (
                4  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 1000
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.hyper_parameter = [1, 0.01]
            # 7 learning rate
            self.D_lr = 1e-3
            self.D_weight_decay = 1e-5
            self.D_step_size = 3
            self.D_gamma = 0.95

        if self.data_name == "USPSdata_20_uni":
            self.train_path = "./datasets/USPSdata_20_uni.mat"
            self.test_path = "./datasets/USPSdata_20_uni.mat"

            self.input_shape = [16, 16, 1]  # image shape
            self.num_classes = 10
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 501
            self.iteration_rate = (
                5  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 2
            self.Batch_size = 200  # 1854

            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.Multiple_AHML = 1  # AHML
            self.Mul_MSN = 0.01  # MSE

        if self.data_name == "mnist_test":
            self.train_path = "./datasets/mnist_test.mat"
            self.test_path = "./datasets/mnist_test.mat"

            # self.is_T = True
            self.is_img_T = True
            self.input_shape = [28, 28, 1]  # image shape
            self.num_classes = 10

            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 501
            self.iteration_rate = (
                4  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 2
            self.Batch_size = 1000
            # 6 hyper-parameter
            self.margin = 0.9  # the margin of d when different
            self.Multiple_AHML = 1  # AHML
            self.Mul_MSN = 0.01  # MSE

        if self.data_name == "imm40":
            self.train_path = "./datasets/imm40.mat"
            self.test_path = "./datasets/imm40.mat"
            # 2.[num , features]
            self.is_T = True
            self.is_img_T = True
            self.input_shape = [32, 32, 1]  # image shape
            self.num_classes = 40

            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 401
            self.iteration_rate = (
                20  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 2
            self.Batch_size = 200  # 240
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.Multiple_AHML = 1  # AHML
            self.Mul_MSN = 0.01  # MSE

        other_data = False
        if other_data:
            # 5 epoch size
            self.Init_Epoch = 0
            self.Unfreeze_Epoch = 401
            self.iteration_rate = (
                10  # Number of iterations of training set and validation set
            )
            self.epoch_size_val = 1
            self.Batch_size = 200
            # 6 hyper-parameter
            self.margin = 0.7  # the margin of d when different
            self.hyper_parameter = [1, 0.01]
            # 7 learning rate
            self.D_lr = 4e-4
            self.D_weight_decay = 1e-4
            self.D_gamma = 0.992

        # ========================  General config =====================================
        # 8.running on device
        self.gpus = "1"  # Number of GPUs used during training
        self.g_id = "0"  # id of GPU
        self.resume = False  # if load the pretrained weights

        self.Multiple_AHML, self.Mul_MSN = self.hyper_parameter

        # ========================  General Advisor config =====================================

        self.hidden1 = 256
        self.hidden2 = 128
        self.Advisor_dropout = 0.1  # 'Dropout rate (1 - keep probability).'
        self.iteration = 64
        self.Advisor_epochs = self.iteration * 256
        self.Advisor_lr = 1e-4
        self.Advisor_step_size = self.iteration * 8
        self.Advisor_gamma = 0.98

        self.my_path = "./logs/{}".format(str(self.data_name))
        self.Phd_path = self.my_path + "/{}/Rho-{}_gama-{}_mse_line2_mu_1000/".format(
            "Phd", str(self.margin), str(self.Mul_MSN)
        )
        self.Advisor_path = self.my_path + "/{}/h1-{}_h2-{}".format(
            "Advisor", self.hidden1, self.hidden2
        )
        # path of saving model
        self.model_path = "./model/{}/".format(self.data_name)
        self.save_path = self.Phd_path


args = [
    "umist_1024",
    "COIL20",
    "Palm",
    "fashion-mnist",
    "USPS",
    "MSRA",
    "USPSdata_20_uni",
    "mnist_test",
    "CIFAR_test",
    "imm40",
    "JAFFE",
]

config = DefaultConfigs(args[2])
