# -*- coding: utf-8 -*-
"""
@Time          : 2020/11/16 18:50
@Author        : BarneyQ
@File          : medical_env.py
@Software      : PyCharm
@Description   : 环境：一张图   520761-8.png 226,132,455,323,1
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
from collections import deque

import cv2
import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.autograd import Variable

INITIAL_EPSILON = 0.0001  # epsilon的初始值
BATCH_SIZE = 32  # 训练批次
GAMMA = 0.99  # decay rate of past observations
UPDATE_TIME = 100  # 每隔UPDATE_TIME轮次，对target网络的参数进行更新
REPLAY_MEMORY = 50000  # 记忆库
OBSERVE = 1000.  # 前OBSERVE轮次，不对网络进行训练，只是收集数据，存到记忆库中
EXPLORE = 2000000.
FINAL_EPSILON = 0.0001  # epsilon的最终值
FRAME_PER_ACTION = 1  # 每隔FRAME_PER_ACTION轮次，就会有epsilon的概率进行探索


class MedicalEnv:
    """
    环境类：
        导入一张图片
    """
    def __init__(self):
        pass

    def reset(self):
        self.img_path = './520761-8.png'
        self.cur_observation = [0, 0, 0, 0]  # 定义状态 [左上角x，y；右下角x，y]
        self.next_observation = [0, 0, 0, 0]  # 下一个状态
        self.true_box = [226, 132, 455, 323]  # 真实box坐标
        self.terminal = False  # 回合结束标志 默认没有
        self.env = cv2.imread(self.img_path)
    #给出初始的状态
    def game_state(self):
        return self.cur_observation

    #输入一个动作(向量)，输出下一个状态、reward、terminal标志
    def step(self,action):
        observation_1 = self.get_next_observation(action)
        reward_0,terminal = self.get_reward()
        return observation_1,reward_0,terminal

    # 辅助函数： 根据输入动作计算下一个状态
    def get_next_observation(self,action):
        # action = [delta_x，delta_y，flatter_x，flatter_y，scale_s ]
        self.next_observation = self.cur_observation.copy()
        # 1. 移动
        self.next_observation[0] += action[0]
        self.next_observation[1] += action[1]
        self.next_observation[2] += action[0]
        self.next_observation[3] += action[1]

        # 2. 单轴放缩
        width =  self.next_observation[2] - self.next_observation[0]
        height =  self.next_observation[3] - self.next_observation[1]
        del_width = width * action[2]
        del_height = height * action[3]
        width = del_width - width
        height = del_height - height
        self.next_observation[2] += width
        self.next_observation[3] += height
        # 3. 右下角放缩
        width = self.next_observation[2] - self.next_observation[0]
        height =  self.next_observation[3] - self.next_observation[1]
        del_width = width * action[4]
        del_height = height * action[4]
        self.next_observation[2] = self.next_observation[0] + del_width
        self.next_observation[3] = self.next_observation[1] + del_height

        # 超范围 <0 >0
        if self.next_observation[0]< 0:
            self.next_observation[0] = 0
        if self.next_observation[1] <0:
            self.next_observation[1] = 0
        if self.next_observation[2] <0:
            self.next_observation[2] = 0
        if self.next_observation[3] <0:
            self.next_observation[3] = 0


        return self.next_observation

    def get_box(self):
        cv2.imshow('box',self.env)
        cv2.waitKey(0)

    #定义 reward
    def get_reward(self):
        cur_iou = self.cpt_iou(self.true_box,self.cur_observation)
        next_iou = self.cpt_iou(self.true_box,self.next_observation)
        del_iou = next_iou - cur_iou
        reward = 0
        if del_iou < 0:
            reward =-1
            self.terminal = True # 结束回合
        else:
            reward =0.01
            self.terminal = False # 继续
        if next_iou >=0.6:
            reward = 1
            self.terminal = True # niubi

        return reward,self.terminal

    def cpt_iou(self,box1,box2):
        x0, y0, x1, y1 = box1
        x2, y2, x3, y3 = box2
        s1 = (x1 - x0) * (y1 - y0)
        s2 = (x3 - x2) * (y3 - y2)
        w = max(0, min(x1, x3) - max(x0, x2))
        h = max(0, min(y1, y3) - max(y0, y2))
        inter = w * h
        iou = inter / (s1 + s2 - inter)
        return iou

# 神经网络结构，结构较为容易理解
class DeepNetWork(nn.Module):
    def __init__(self, ):
        super(DeepNetWork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.out(x)

class BrainDQNMain:
    def save(self):
        print("save model parm")
        torch.save(self.Q_net.state_dict(), 'params3.pth')

    def load(self):
        if os.path.exists("params3.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('params3.pth'))
            self.Q_netT.load_state_dict(torch.load('params3.pth'))

    def __init__(self, actions):
        # 在每个timestep下agent与环境交互得到的转移样本 (st,at,rt,st+1) 储存到回放记忆库，
        # 要训练时就随机拿出一些（minibatch）数据来训练，打乱其中的相关性
        self.replayMemory = deque()  # init some parameters
        self.timeStep = 0
        # 有epsilon的概率，随机选择一个动作，1-epsilon的概率通过网络输出的Q（max）值选择动作
        self.epsilon = INITIAL_EPSILON
        # 初始化动作
        self.actions = actions
        # 当前值网络
        self.Q_net = DeepNetWork()
        # 目标值网络
        self.Q_netT = DeepNetWork()
        # 加载训练好的模型，在训练的模型基础上继续训练
        self.load()
        # 使用均方误差作为损失函数
        self.loss_func = nn.MSELoss()
        LR = 1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    # 使用minibatch训练网络
    def train(self):  # Step 1: obtain random minibatch from replay memory
        # 从记忆库中随机获得BATCH_SIZE个数据进行训练
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]  # Step 2: calculate y
        # y_batch用来存储reward
        y_batch = np.zeros([BATCH_SIZE, 1])
        nextState_batch = np.array(nextState_batch)  # print("train next state shape")
        # print(nextState_batch.shape)
        nextState_batch = torch.Tensor(nextState_batch)
        action_batch = np.array(action_batch)
        # 每个action包含两个元素的数组，数组必定是一个1，一个0，最大值的下标也就是该action的下标
        index = action_batch.argmax(axis=1)
        print("action " + str(index))
        index = np.reshape(index, [BATCH_SIZE, 1])
        # 预测的动作的下标
        action_batch_tensor = torch.LongTensor(index)
        # 使用target网络，预测nextState_batch的动作
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch = QValue_batch.detach().numpy()
        # 计算每个state的reward
        for i in range(0, BATCH_SIZE):
            # terminal是结束标志
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                # 网络的输出层的维度为2，将输出值中的最大值作为Q值
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])
        state_batch_tensor = Variable(torch.Tensor(state_batch))
        y_batch_tensor = Variable(torch.Tensor(y_batch))
        y_predict = self.Q_net(state_batch_tensor).gather(1, action_batch_tensor)
        loss = self.loss_func(y_predict, y_batch_tensor)
        print("loss is " + str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每隔UPDATE_TIME轮次，用训练的网络的参数来更新target网络的参数
        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    # 更新记忆库，若轮次达到一定要求则对网络进行训练
    def setPerception(self, nextObservation, action, reward, terminal):  # print(nextObservation.shape)
        # 每个state由4帧图像组成
        # nextObservation是新的一帧图像,记做5。currentState包含4帧图像[1,2,3,4]，则newState将变成[2,3,4,5]
        newState = np.append(self.currentState[1:, :, :], nextObservation,
                             axis=0)  # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # 将当前状态存入记忆库
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        # 若记忆库已满，替换出最早进入记忆库的数据
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        # 在训练之前，需要先观察OBSERVE轮次的数据，经过收集OBSERVE轮次的数据之后，开始训练网络
        if self.timeStep > OBSERVE:  # Train the network
            self.train()

        # print info
        state = ""
        # 在前OBSERVE轮中，不对网络进行训练，相当于对记忆库replayMemory进行填充数据
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1

# 获得下一步要执行的动作
    def getAction(self):
        currentState = torch.Tensor([self.currentState])
        # QValue为网络预测的动作
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        # FRAME_PER_ACTION=1表示每一步都有可能进行探索
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:  # 有epsilon得概率随机选取一个动作
                action_index = random.randrange(self.actions)
                print("choose random action " + str(action_index))
                action[action_index] = 1
            else:  # 1-epsilon的概率通过神经网络选取下一个动作
                action_index = np.argmax(QValue.detach().numpy())
                print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        else:  # 程序貌似不会走到这里
            action[0] = 1  # do nothing

        # 随着迭代次数增加，逐渐减小episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    # 初始化状态
    def setInitState(self, observation):
        # 增加一个维度，observation的维度是80x80+4，讲过stack()操作之后，变成4x(80x80+4)
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)
        print(self.currentState.shape)




if __name__ == '__main__':
    me = MedicalEnv()  #加载环境
    # action = [delta_x，delta_y，flatter_x，flatter_y，scale_s ]
    action = [0,0,1,1,1]  # 初始动作



    next_observation = me.step(action)
    print(next_observation)
    print(me.cur_observation)