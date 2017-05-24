import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import namedtuple
from torch.autograd import Variable
from ..proj_utils.torch_utils import *
import torch.optim as optim


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        # if shared_param.grad is not None:
        #    return
        shared_param._grad = param.grad


def get_rewards(refs, gts):
    """
    refs: as the list
    gts: also a list of list.
    """
    return refs - gts


class ACModel(nn.Module):
    """
    This class takes perform one step prediction based on input.
    """

    def __init__(self, word_num, emb_dim=128, cont_dim=256, feat_dim=256,
                 lstm_dim=256, actor_hid_dim=128,
                 critic_hid_dim=128):
        self.__dict__.update(locals())

        self.gate_dim = self.feat_dim + self.lstm_dim + self.cont_dim  # use x2cont, h, att to decide
        self.lstm_inp_dim = self.cont_dim + self.feat_dim

        super(ACModel, self).__init__()
        self.train()

        outputs = namedtuple('output', 'done states')

        # self.emb_model = nn.Embedding(self.word_num, self.emb_dim)
        self.one_hot = one_hot(self.word_num)
        self.x2cont = nn.Linear(self.word_num, self.cont_dim)
        self.h2cont = nn.Linear(self.lstm_dim, self.cont_dim)

        self.inp2hid = nn.Linear(self.feat_dim, self.lstm_dim)  # only used for get init hidden from fc
        self.inp2cell = nn.Linear(self.feat_dim, self.lstm_dim)  # only used for get init cell from fc
        self.inp2start = nn.Linear(self.feat_dim, self.word_num)  # only used for get init start x from fc

        self.cont2key = nn.Linear(self.cont_dim, self.feat_dim)
        self.cont2strength = nn.Linear(self.cont_dim, 1)

        self.att_gates = nn.Linear(self.gate_dim, 1)

        self.lstm_1 = torch.nn.LSTMCell(self.lstm_inp_dim, self.lstm_dim, bias=True)  # take x and

        self.actor_hid = nn.Linear(self.lstm_dim, self.actor_hid_dim)
        self.critic_hid = nn.Linear(self.lstm_dim, self.critic_hid_dim)

        self.actor = nn.Linear(self.actor_hid_dim, self.word_num)
        self.critic = nn.Linear(self.critic_hid_dim, 1)

        # self.long_mem =

    def get_states(self, fc):
        h = F.tanh(self.inp2hid(fc))
        c = F.tanh(self.inp2cell(fc))
        # h = Variable(self.W_lstm.data.new(batch_size, self.nhid).fill_(0.0), requires_grad = True)
        # c = Variable(self.W_lstm.data.new(batch_size, self.nhid).fill_(0.0), requires_grad = True)
        return (h, c)

    def update_long_memory(self, writting_key, writting_content):
        """
        Long memory should act like weigths, but it is updated in a special way.
        Long memory is not even a variable. It should be stated as Variable at forward to cut the history.

        attended memory should be used to predict gate for guiding whether or not to use it.
        when we update the long memory, we don't want it to be connected to other sample, 
        meaning it is only the leaf node.
        otherwise, the history will soon blow up. 
        easy way to do this is: make the writting as tensor. 
        It also should be protected. 
        """
        pass

    def forward(self, inps, args):
        """
        word_emb --> inp_encode_dim 
        atten ---->  inp_encode_dim 

        x is the last prediction word emb transformed to proper hidden states.
        x should be ready to be add to 
        x + (map others into this preact map, combined with attention and memory)
        we need a long term buffer as experience memory, a episode working memory

        Parameters
        ----------
        inps:  
            x :   the word embedding from last run.
            feat: batch * num_slot * dim
            states: h_tm1, c_tm1
            args: 
        """
        h_tm1, c_tm1 = inps.states
        # we need to use x, and h_tm1 to get the retrieval key first, and strenght
        x_cont = self.x2cont(inps.x)
        h_cont = self.h2cont(h_tm1)
        control_tmp = F.relu(x_cont + h_cont)
        key = F.tanh(self.cont2key(control_tmp))
        strength = F.relu(self.cont2strength(control_tmp)) + 1.0

        alpha = get_content_address(inps.feat, expand_dims(key, -1), expand_dims(strength, -1))
        cand_att_vec = update_read_vectors(inps.feat, alpha)

        gates_att = F.sigmoid(self.att_gates(torch.cat([cand_att_vec, h, x], 1)))
        att_vec = cand_att_vec * gates_att.expand_as(cand_att_vec)

        inp_lstm = torch.cat([att_vec, x_cont], 1)

        h, c = self.lstm_1(inp_lstm, (h_tm1, c_tm1))

        actor_hidden = F.relu(self.actor_hid(h))
        critic_hidden = F.relu(self.critic_hid(h))

        action_logit = self.actor(actor_hidden)
        value = self.critic(critic_hidden)

        return action_logit, value, (h, c)

def is_done(action):
    return action == 0


def one_hot(nwords):
    eye_var = torch.eye


def train(generator, cnn_model, shared_model, args):
    """
    Parameter
    ---------
    generator: flow of data and sentence. This might need to be a class 
               that is thread safe, since we need multi-process
            data: (1, row, col, chn)
            sentence: (steps, emb_dim)
    cnn_model:
            input:  data
            output: 
                    fc_feat (1, fc_len)
                    feat    (1, row, col, feat_dim)
    args:   system args parameters

    """
    cnn_model.eval()
    ac_model = ACModel(word_num=56)
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    inputs_tuple = namedtuple('inputs', 'x feat states args')

    # Each sentence is an environment

    for Img_data, sentence_int in generator:
        # Img_data: (nsample, channel, row, col)
        # sentence: Tensor (maxlen, nsample)
        sentence = ac_model.one_hot(sentence_int) #(maxlen, nsample, emb_dim)
        fc_feat, feat = cnn_model.get_feature(Img_data)
        fc_feat = fc_feat[0]
        feat = feat[0]

        h_lstm, c_lstm = ac_model.get_states(fc_feat)
        states = (h_lstm, c_lstm)

        word_emd = ac_model.inp2start(fc_feat)

        intpus = inputs_tuple([word_emd, feat, states, args])
        timesteps = sentence.size()[0]
        step = 0
        # for one traning sample, however, we don't follow the sentence length
        while True:
            entropies = []
            rewards = []
            log_probs = []
            values = []
            refs = []

            for _ in range(args.num_steps):
                step += 1
                logit, val, (h, c) = ac_model(intpus)
                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy_action = -(log_prob * prob).sum(1)
                entropies.append(entropy_action)

                # action = prob.multinomial()
                action = multinomial(prob)
                refs.append(action)
                res_log_prob = torch.gather(log_prob, 1, action)
                # res_log_prob = log_prob.gather(1, action)
                log_probs.append(res_log_prob)

                word_emd = ac_model.emb_model[action]
                states = (h, c)
                intpus = inputs_tuple([word_emd, feat, states, args])

                # we stop and learn every arg.num_step.
                done = is_done(action)
                outof_step = step == args.max_steps

                values.append(val)
                if done or outof_step:
                    reward = get_rewards(refs, sentence)
                    if outof_step:
                        reward = reward - 1  # penalty for not stopping before max_steps
                    rewards.append(reward)
                else:
                    rewards.append(0)  # no explicit rewards for each step.

                if done or outof_step:
                    break

            val = Variable(torch.zeros(1, 1))
            if not (done or outof_step):
                _, val, _ = ac_model(intpus)  # we just evaluate for this time.

            values.append(val)

            value_loss = 0
            policy_loss = 0
            gae = Variable(torch.zeros(1, 1))
            disR = val
            # reward: 0   0   0    0   0
            # val:    0.1 0.2 0.4  0.6 0.7 1
            for i in reversed(range(len(rewards))):
                disR = args.gamma * disR + rewards[i]
                adv = disR - values[i]
                value_loss = value_loss + 0.5 * adv.pow(2)
                # generalized adv estimation
                delta_t = rewards[i] + args.gamma * \
                                       values[i + 1] - values[i]
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]

            optimizer.zero_grad()
            total_loss = (policy_loss + value_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(ac_model.parameters(), 40)
            ensure_shared_grads(ac_model, shared_model)
            optimizer.step()

