import argparse
import random
import gym
import torch
from torch.optim import Adam
from tester import Tester
from buffer import ReplayBuffer
from config import Config
from core.util import get_class_attr_val
from model import DQN
from trainer import Trainer

class DDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)
        #ReplayBuffer 함수는 buffer.py에 존재하며 일정 이상의 수용력을 넘길 시 저장한다.

        self.model = DQN(self.config.state_dim, self.config.action_dim).cuda()
        #model을 DQN으로 정한다.
        self.target_model = DQN(self.config.state_dim, self.config.action_dim).cuda()
        #target_model을 self.model와 같은 것을 지정받아 저장한다.
        self.target_model.load_state_dict(self.model.state_dict())
        #state_dict 함수는 학습 네트워크의 모든 학습 가능한 매개변수들을 OrdereDict형식으로 변경한다. 이후 load_state_dict 함수로 모든 매개변수들을 학습 네트워크의 매개변수 값으로 업데이트 한다.
        # 이 때 ordereDict는 삽입된 순서를 기억하는 딕셔너리 자료형이다.
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)
        #파이토치를 이용하여 학습할 때 최적화 알고리즘인 Adam을 이용하여 학습 네트워크의 가중치를 업데이트 한다.
        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        #만약 epsilon 에 아무 값이 없다면, epsilon에는 min 값을 넣는다.
        if random.random() > epsilon or not self.is_training:
        #if A or not B는 변수 A가 참이거나 B가 거짓일 때를 의미한다. 이 곳에서는 random > epsilon이거나 training 중이 아닐 시에 if 문에 들어간다.
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        #state 변수를 Float 자료형의 Tensor로 변환하고, 차원을 추가해 주었다. 이 때 unsqueeze 함수는 텐서의 차원을 추가하는 것으로 0을 인자로 넘겨주어 텐서의 가장 왼쪽에 차원을 추가한다.
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
            #가장 큰 값 선택. q_value.max(1)[1]은 Q-value tensor 에서 가장 큰 값을 가진 index 반환
            #max()의 함수의 dim 매개변수가 1로 설정되어 있어 각 행마다 최대값을 가진 인덱스가 반환. 이후, [1]을 사용하여 해당 인덱스 추출됨. 
        else:
            action = random.randrange(self.config.action_dim)
            # cuda 를 사용하지 않는 다면 랜덤성을 지니게 됨.
        return action
        #Q_value를 state 에 대해서 계산하고, 그 중 가장 큰 값을 가진 action을 선택하게 된다. 

    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)
        #초기 값들을 랜덤으로 배치한다.
        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)
        #변수들을 torch.tensor 차원으로 선언 해 줌

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()
        next_q_state_values = self.target_model(s1).cuda()
        #현재 학습하고 있는 모델에 s0과 s1을 넣는다. 이를 통해 Q-value에 대해서 계산한다. .cuda는 GPU에서 수행하도록 해 주는 코드이다.
        #next Q value 와 state value 에는 s1을 입력으로 받아 다음 상태에 대한 Q value에 대해서 계산하게 된다. 
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        #gather 함수는 wide format을 long format로 바꿔 주는 것. 여기서 wide는 세로형, long은 가로형이라고 의미한다. -> s0가 주어졌을 때의 action a가 unsqueeze에 저장된다.
        #즉, 이 활동을 통해 q_value에는 s0에서의 행동 a에 대한 Q-value 가 담기게 된다.
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        #next_q_value에는 s0에서 행동에 대한 Q-value를 지니고 있다. 이 떄, 가장 높은 Q-value를 가지는 행동의 인덱스와 그 값을 추출하게 된다.
        #max(1)[1]은 각 샘플마다 최대 Q-value 를 가지는 행동의 인덱스를 나타내는 tensor이다.
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        #reward + gamma*maxQ-value*(1-done)이 expected_q_value가 된다. 이 때 gamma는 감가율(discount fator) 이다.
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        #loss는 손실된 것이 얼마나 있는 지에 대해서 알아보는 함수이다. q_value에 앞서 계산한 예상 Q_value를 빼서 이에 대한 손실을 확인한다. 이 손실을 줄이는 방식으로 학습한다.
        #.detach() : gradient가 현재 계산되지 않도록 한다.
        self.model_optim.zero_grad()
        #adam을 이용하여 최적화를 진행한다.
        loss.backward()
        #이 곳에서는 손실함수의 증가율에 대해서 알아볼 수 있다. -> loss에 대한 증가율이 계산되어 각 파라미터의 grad 속성에 저장된다. 이후 .step을 통하여 저장한다.
        self.model_optim.step()
        # loss.backward 메서드 호출 이후, 최적화를 이용해 네트워크의 파라미터를 업데이트 하는 단계
        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        #일정한 주기 마다 현재 모델의 가중치를 타겟 모델의 가중치로 업데이트 한다.

        return loss.item()
        #계산된 손실(loss) 반환. 스칼라 값으로 반환
    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    #description을 ''으로 지정해 준 것이다. 
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='CartPole-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()
    # ddqn.py --train --env CartPole-v0

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 500
    config.frames = 160000
    config.use_cuda = True
    config.learning_rate = 1e-3
    config.max_buff = 1000
    #max_buff 이상의 학습 량을 가지게 된다면 self.buffer 에 append 하고, 다음 학습으로 넘어간다.
    config.update_tar_interval = 100
    config.batch_size = 128
    config.print_interval = 200
    config.log_interval = 200
    config.win_reward = 198     # CartPole-v0
    config.win_break = True

    env = gym.make(config.env)
    config.action_dim = env.action_space.n
    config.state_dim = env.observation_space.shape[0]
    agent = DDQNAgent(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test()