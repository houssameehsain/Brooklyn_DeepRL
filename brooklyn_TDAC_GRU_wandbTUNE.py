import numpy as np
import socket
import cv2

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import wandb

# Define Socket
HOST = '127.0.0.1'
timeout = 20

def done_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    return_str = return_byt.decode() 

    return eval(return_str)

def reward_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    return_str = return_byt.decode()
    return_int = int(return_str) 

    return return_int

def fp_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    fp = return_byt.decode()

    return fp

def send_ep_count_to_gh_client(socket, message):
    message_str = str(message)
    message_byt = message_str.encode()

    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(message_byt)

def send_to_gh_client(socket, message):
    message_str = ''
    for item in message:
        listToStr = ' '.join(map(str, item))
        message_str = message_str + listToStr + '\n'

    message_byt = message_str.encode()
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(message_byt)   

# Set device
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(f'Used Device: {device}')

img_size = 256
def read_obs(fp):
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_arr = np.array(im)
    im_arr = im_arr.reshape((3, img_size, img_size))
    im_arr = im_arr / 255.0
    state = torch.from_numpy(im_arr).type(torch.float32)

    return state

# Actor Critic Model Architecture 
def enc_block(in_c, out_c, BN=True):
    if BN:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return conv
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return conv
class GRUpolicy(nn.Module):
    def __init__(self, n_gru_layers):
        super(GRUpolicy, self).__init__()

        #critic
        self.critic_enc1 = enc_block(3, 32, BN=False)
        self.critic_enc2 = enc_block(32, 64, BN=True)
        self.critic_enc3 = enc_block(64, 128, BN=True)
        self.critic_enc4 = enc_block(128, 128, BN=True)

        self.critic_linear1 = nn.Linear(512, 256)
        self.critic_linear2 = nn.Linear(256, 128)
        self.critic_linear3 = nn.Linear(128, 1)

        # actor
        self.param1_space = torch.from_numpy(np.linspace(start=456, stop=835, num=250))
        self.param2_space = torch.from_numpy(np.linspace(start=325, stop=886, num=250))
        self.param3_space = torch.from_numpy(np.linspace(start=0, stop=180, num=250))
        self.param4_space = torch.from_numpy(np.linspace(start=20, stop=70, num=250))
        self.param5_space = torch.from_numpy(np.linspace(start=20, stop=70, num=250))
        self.param6_space = torch.from_numpy(np.linspace(start=10, stop=200, num=250))

        self.gru1 = nn.GRU(4, 128, n_gru_layers, batch_first=True)
        self.actor1_linear = nn.Linear(128, 250)

        self.gru2 = nn.GRU(4, 128, n_gru_layers, batch_first=True)
        self.actor2_linear = nn.Linear(128, 250)

        self.gru3 = nn.GRU(4, 128, n_gru_layers, batch_first=True)
        self.actor3_linear = nn.Linear(128, 250)

        self.gru4 = nn.GRU(4, 128, n_gru_layers, batch_first=True)
        self.actor4_linear = nn.Linear(128, 250)

        self.gru5 = nn.GRU(4, 128, n_gru_layers, batch_first=True)
        self.actor5_linear = nn.Linear(128, 250)

        self.gru6 = nn.GRU(4, 128, n_gru_layers, batch_first=True)
        self.actor6_linear = nn.Linear(128, 250)
    
    def forward(self, state):
        state = Variable(state.unsqueeze(0))

        # critic
        enc = self.critic_enc1(state)
        enc = self.critic_enc2(enc)
        enc = self.critic_enc3(enc)
        enc = self.critic_enc4(enc)

        value = F.relu(self.critic_linear1(torch.flatten(enc)))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        # actor
        seq = torch.reshape(enc, (1, 128, 4))

        out1, h_1 = self.gru1(seq)
        out_s1 = torch.squeeze(out1[:, -1, :])
        out_l1 = self.actor1_linear(out_s1)
        p_dist1 = F.softmax(out_l1, dim=-1)
        param1_idx = torch.multinomial(p_dist1, 1)
        param1 = self.param1_space[param1_idx]

        out2, h_2 = self.gru2(seq, h_1) 
        out_s2 = torch.squeeze(out2[:, -1, :])
        out_l2 = self.actor2_linear(out_s2)
        p_dist2 = F.softmax(out_l2, dim=-1)
        param2_idx = torch.multinomial(p_dist2, 1)
        param2 = self.param2_space[param2_idx]

        out3, h_3 = self.gru3(seq, h_2)
        out_s3 = torch.squeeze(out3[:, -1, :])
        out_l3 = self.actor3_linear(out_s3)
        p_dist3 = F.softmax(out_l3, dim=-1)
        param3_idx = torch.multinomial(p_dist3, 1)
        param3 = self.param3_space[param3_idx]

        out4, h_4 = self.gru4(seq, h_3)
        out_s4 = torch.squeeze(out4[:, -1, :])
        out_l4 = self.actor4_linear(out_s4)
        p_dist4 = F.softmax(out_l4, dim=-1)
        param4_idx = torch.multinomial(p_dist4, 1)
        param4 = self.param4_space[param4_idx]

        out5, h_5 = self.gru5(seq, h_4)
        out_s5 = torch.squeeze(out5[:, -1, :])
        out_l5 = self.actor5_linear(out_s5)
        p_dist5 = F.softmax(out_l5, dim=-1)
        param5_idx = torch.multinomial(p_dist5, 1)
        param5 = self.param5_space[param5_idx]

        out6, _ = self.gru6(seq, h_5)
        out_s6 = torch.squeeze(out6[:, -1, :])
        out_l6 = self.actor6_linear(out_s6)
        p_dist6 = F.softmax(out_l6, dim=-1)
        param6_idx = torch.multinomial(p_dist6, 1)
        param6 = self.param6_space[param6_idx]

        # policy_dist = p_dist1 * p_dist2 * p_dist3 * p_dist4 * p_dist5 * p_dist6
        p1 = p_dist1[param1_idx]
        p2 = p_dist2[param2_idx]
        p3 = p_dist3[param3_idx]
        p4 = p_dist4[param4_idx]
        p5 = p_dist5[param5_idx]
        p6 = p_dist6[param6_idx]

        prob = p1 * p2 * p3 * p4 * p5 * p6
        log_prob = torch.log(prob)

        smoothed_entropy = -(torch.sum(p_dist1*torch.log(p_dist1)) +
                            torch.sum(p_dist2*torch.log(p_dist2)) +
                            torch.sum(p_dist3*torch.log(p_dist3)) +
                            torch.sum(p_dist4*torch.log(p_dist4)) +
                            torch.sum(p_dist5*torch.log(p_dist5)) +
                            torch.sum(p_dist6*torch.log(p_dist6)))

        # Hcrude = -(torch.log(p_dist1[param1_idx]) + 
        #             torch.log(p_dist2[param2_idx]) +
        #             torch.log(p_dist3[param3_idx]) +
        #             torch.log(p_dist4[param4_idx]) +
        #             torch.log(p_dist5[param5_idx]) +
        #             torch.log(p_dist6[param6_idx]))

        return value, log_prob, smoothed_entropy, param1, param2, param3, param4, param5, param6

def train():
    # hyperparameters
    hyperparameters = dict(n_steps = 25,
                        n_episodes = 1000000,
                        gamma = 0.99,
                        beta = 0.001,
                        lr = 3e-4,
                        lr_decay = 0.1,
                        n_gru_layers = 1)

    wandb.init(config=hyperparameters, entity='hehsain', project='brooklyn_TDAC_GRU')
    # Save model inputs and hyperparameters
    config = wandb.config
    
    # Initialize DRL model
    actorcritic = GRUpolicy(config.n_gru_layers).to(device)
    ac_optimizer = optim.Adam(actorcritic.parameters(), lr=config.lr, weight_decay = 1e-6)

    # Log gradients and model parameters wandb
    wandb.watch(actorcritic, log="all", log_freq=10)

    all_lengths = []
    average_lengths = []
    all_rewards = []

    for episode in range(config.n_episodes):
        fps = []
        param1L, param2L, param3L, param4L, param5L, param6L = [], [], [], [], [], []
        log_probs = []
        values = []
        rewards = []
        entropy = []

        if episode == 0:
            print('\nStart Loop in GH Client...\n')

        for steps in range(config.n_steps):
            if steps == 0:
                fp = 'D:/RLinGUD/brooklyn_a2c_states/observation_iter0.png'
            else:
                fp = fps[-1]
            
            # Get observation from Memory
            state = read_obs(fp).to(device)
            value, log_prob, smoothed_entropy, param1, param2, param3, param4, param5, param6 = actorcritic.forward(state) 
            value = value.cpu()
            value = value.detach().numpy()[0]

            param1L.append(param1.item())
            param2L.append(param2.item())
            param3L.append(param3.item())
            param4L.append(param4.item())
            param5L.append(param5.item())
            param6L.append(param6.item())

            action = [param1L, param2L, param3L, param4L, param5L, param6L]

            # Send action through socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8080))
                s.settimeout(timeout)
                send_to_gh_client(s, action)

            # Send episode count through socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8083))
                s.settimeout(timeout)
                send_ep_count_to_gh_client(s, episode)

            ######### In between GH script #########################################################

            # Recieve observation file path from gh Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8084))
                s.settimeout(timeout)
                fp = fp_from_gh_client(s)

            # Recieve Reward from gh Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8081))
                s.settimeout(timeout)
                reward = reward_from_gh_client(s)

            # Recieve done from Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8082))
                s.settimeout(timeout)
                done = done_from_gh_client(s)

            fps.append(fp)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy.append(smoothed_entropy) 
            
            if done or steps == config.n_steps-1:
                Qval = 0
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps + 1)
                average_lengths.append(np.mean(all_lengths))

                print(f"episode: {episode}, eps_reward: {np.sum(rewards)}, total length: {steps + 1}, average length: {average_lengths[-1]}")
                break
        
        # compute loss functions
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + config.gamma * Qval
            Qvals[t] = Qval
        
        values = torch.FloatTensor(values).to(device)
        Qvals = torch.FloatTensor(Qvals).to(device)
        log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropy)
        
        advantage = Qvals - values
        actor_loss = -(advantage * log_probs).sum() - config.beta * entropy.sum()
        critic_loss = 0.5 * advantage.pow(2).mean() 
        ac_loss = actor_loss + critic_loss 

        # update actor critic
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
        
        print(f"episode: {episode}, actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}, ac_loss: {ac_loss.item()} \n")

        # Log metrics to visualize performance wandb
        wandb.log({'episode': episode, 'reward': np.sum(rewards), 'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'ac_loss': ac_loss.item()})


if __name__ == "__main__":
    # Log in to W&B account
    wandb.login(key='a62c193ea97080a59a7f646248cd9ec23346c61c')

    sweep = False
    if sweep:
        sweep_config = {
                'method': 'grid', #grid, random
                'metric': {
                'name': 'ac_loss',
                'goal': 'minimize'   
                },
                'parameters': {
                    'lr': {
                        'values': [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
                    },
                    'n_gru_layers':{
                        'values':[1, 2]
                    }
                }
            }

        sweep_id = wandb.sweep(sweep_config, project='brooklyn_TDAC_GRU')

        wandb.agent(sweep_id, train)

    else:
        train()