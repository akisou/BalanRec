import collections
import json
import os.path
import sys
import copy
import pandas as pd
import numpy as np
import torch
from random import sample
from tqdm import tqdm
from torch import nn
from sklearn.cluster import OPTICS
from BalanRec.model.init import xavier_normal_initialization
from BalanRec.model.loss import EmbLoss
from BalanRec.model.general_recommender.MultiHeadAtt import MultiHeadAtt
from BalanRec.model.abstract_recommender import GeneralRecommender
from BalanRec.utils import InputType
from BalanRec.model.general_recommender.BalanRecUserSide import BalanRecUserSide
sys.path.append('.')

class BalanRecItemSide(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BalanRecItemSide, self).__init__(config, dataset)
        self.embed_size = 768
        self.in_size = config['in_size']
        self.hid_size = config['hid_size']
        self.add_self_att_on = config['add_self_att_on']
        self.dialog_to_use_num = config['dialog_to_use_num']
        self.query_to_user_num = config['query_to_user_num']
        self.dialog_cluster = config['dialog_cluster']
        self.black_list_using = config['black_list_using']
        self.patient_module = config['patient_module']
        self.PROFILE_ID = config['PROFILE_FIELD']
        self.QUERY_ID = config['QUERY_FIELD']
        self.DIALOG_ID = config['DIALOG_FIELD']

        self.txt_embeddings = dataset.txt_embeddings
        self.txt_embeddings['q'][0] = [0] * self.embed_size
        self.txt_embeddings['profile'][0] = [0] * self.embed_size
        self.txt_embeddings['dialogue'][0] = [0] * self.embed_size
        self.cluster_emb_path = config.final_config_dict['data_path'] + '/cluster_emb.json'
        self.nega_q_embeddings = dataset.nega_q_embeddings
        self.item2nega_user = dataset.item2nega_user
        if type(self.item2nega_user) != int:
            self.item2nega_user[0] = []
        self.inter = pd.DataFrame({
            self.USER_ID: dataset.inter_feat[self.USER_ID].tolist(),
            self.ITEM_ID: dataset.inter_feat[self.ITEM_ID].tolist(),
            config['RATING_FIELD']: dataset.inter_feat[config['RATING_FIELD']].tolist()
        })
        self.items = dataset.item_feat[self.ITEM_ID].tolist()
        self.item2patient_emb = self.cluster_dialogue_of_items()
        self.device = config["device"]
        if self.add_self_att_on == "profile":
            # print("Model: MUL-ATT W/O D")
            self.goodat_attention = MultiHeadAtt(config['head_num'], config['dropout'])
        elif self.add_self_att_on == "dialogs":
            # print("Model: MUL-ATT W/O P")
            self.dialog_attention = MultiHeadAtt(config['head_num'], config['dropout'])
        else:
            # print("Model: MUL-ATT FULL")
            self.attention = MultiHeadAtt(config['head_num'], config['dropout'])

        if self.black_list_using:
            self.attention_black_list = MultiHeadAtt(config['head_num'], config['dropout'])

        # define function
        self.reg_weight = config["reg_weight"]
        self.black_list_weight = config['black_list_weight']
        self.user_side_weight = config['user_side_weight']
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=self.hid_size)
        self.fc1 = nn.Linear(self.in_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

        # BalanRec user part model initial
        self.UserSide_model = BalanRecUserSide(config, dataset)
        
    def forward(self, user, item):
        profile = torch.FloatTensor([self.txt_embeddings[self.PROFILE_ID][elem.tolist()] for elem in item])\
            .to(self.device)
        profile = profile.reshape((profile.shape[0], 1, profile.shape[-1]))
        dialogs = self.pack_dialogs(item).to(self.device)
        query = torch.FloatTensor([self.txt_embeddings[self.QUERY_ID][elem.tolist()] for elem in user]).to(self.device)

        # our ablations: MUL-ATT (W/O D or W/O P)
        if self.add_self_att_on == "profile":
            q, k, v = copy.deepcopy(profile), copy.deepcopy(profile), copy.deepcopy(profile)
            dr_emb, _ = self.goodat_attention(q, k, v)
        elif self.add_self_att_on == "dialogs":
            q, k, v = copy.deepcopy(dialogs), copy.deepcopy(dialogs), copy.deepcopy(dialogs)
            dr_emb, _ = self.dialog_attention(q, k, v)
            dr_emb = torch.mean(dr_emb, dim=1).unsqueeze(1)
        else: # our model, MUT-ATT (FULL)
            q, k, v = copy.deepcopy(profile), copy.deepcopy(dialogs), copy.deepcopy(dialogs)
            dr_emb, _ = self.attention(q, k, v)

        dr_emb = dr_emb.reshape((dr_emb.shape[0], dr_emb.shape[-1]))

        # black list module
        if self.black_list_using:
            query_bl = self.pack_dialogs_black_list(item).to(self.device)
            q_bl, k_bl, v_bl = copy.deepcopy(query_bl), copy.deepcopy(query_bl), copy.deepcopy(query_bl)
            dr_emb_bl, _ = self.attention_black_list(q_bl, k_bl, v_bl)
            dr_emb_bl = torch.mean(dr_emb_bl, dim=1)
            return query, dr_emb, dr_emb_bl
        else:
            return query, dr_emb

    def cluster_dialogue_of_items(self):
        if self.dialog_cluster:
            # if existing, load it
            # if os.path.exists(self.cluster_emb_path):
            #     item2patients_emb = json.load(open(self.cluster_emb_path))
            #     return item2patients_emb
            item2patients_emb = dict()
            cluster_method = OPTICS(min_samples=2,)  # metric='euclidean'
            for item in tqdm(self.items):
                patients = self.inter[self.inter[self.ITEM_ID] == item][self.USER_ID].values
                candidate_patis = [patient for patient in patients
                                   if patient in self.txt_embeddings[self.DIALOG_ID].keys()]

                # item2patients_emb[item] = [self.txt_embeddings[self.DIALOG_ID][elem] for elem in candidate_patis]

                # cluster module (balance the topics size of dialogues)
                # cluster_method = DBSCAN(eps=3, min_samples=1  0)
                if len(candidate_patis) >= 4:
                    txt_emb_data = [self.txt_embeddings[self.DIALOG_ID][elem] for elem in candidate_patis]
                    cluster_results = cluster_method.fit(txt_emb_data)
                    box = collections.defaultdict(list)
                    labels = cluster_results.labels_
                    ordering = cluster_results.ordering_
                    for i in range(len(labels)):
                        box[labels[i]].append(txt_emb_data[ordering[i]])
                    item2patients_emb[item] = [list(np.mean(elem, axis=0)) for elem in list(box.values())]
                else:
                    item2patients_emb[item] = [self.txt_embeddings[self.DIALOG_ID][elem] for elem in
                                               candidate_patis]

            # json.dump(item2patients_emb, open(self.cluster_emb_path, 'w'))
        else:
            item2patients_emb = dict()
            for item in tqdm(self.items):
                patients = self.inter[self.inter[self.ITEM_ID] == item][self.USER_ID].values
                candidate_patis = [patient for patient in patients
                                   if patient in self.txt_embeddings[self.DIALOG_ID].keys()]

                item2patients_emb[item] = [self.txt_embeddings[self.DIALOG_ID][elem] for elem in candidate_patis]

        return item2patients_emb

    def pack_dialogs(self, item):
        dialogs = []
        for elem in item.tolist():
            candidate_patis = self.item2patient_emb[elem]
            # sample and pad
            if len(candidate_patis) > self.dialog_to_use_num:
                sample_candidate_patis = sample(candidate_patis, self.dialog_to_use_num)
                sub_dialogs = sample_candidate_patis
            else:
                pad_size = self.dialog_to_use_num - len(candidate_patis)
                sub_dialogs = candidate_patis
                sub_dialogs.extend([[0] * self.embed_size] * pad_size)
            dialogs.append(sub_dialogs)

        return torch.FloatTensor(dialogs)

    def pack_dialogs_black_list(self, item):
        patients_of_doctors = [self.item2nega_user[elem.tolist()] for elem in item]
        dialogs = []
        for patients in patients_of_doctors:
            candidate_patis = [self.nega_q_embeddings[str(patient)] for patient in patients
                               if str(patient) in self.nega_q_embeddings.keys()]

            # sample and pad
            if len(candidate_patis) > self.query_to_user_num:
                sample_candidate_patis = sample(candidate_patis, self.query_to_user_num)
                sub_dialogs = sample_candidate_patis
            else:
                pad_size = self.query_to_user_num - len(candidate_patis)
                sub_dialogs = candidate_patis
                sub_dialogs.extend([[0] * self.embed_size] * pad_size)
            dialogs.append(sub_dialogs)

        return torch.FloatTensor(dialogs)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # item side
        if self.black_list_using:
            user_e, pos_item_e, pos_item_bl_e = self.forward(user, pos_item)
            user_e, neg_item_e, neg_item_bl_e = self.forward(user, neg_item)
            # pos_item_bl_loss = torch.sum(torch.mul(user_e, pos_item_bl_e))
            neg_item_bl_loss = torch.mean(self.sigmoid(torch.mul(user_e, neg_item_bl_e)))
            black_list_loss = neg_item_bl_loss
        else:
            user_e, pos_item_e = self.forward(user, pos_item)
            user_e, neg_item_e = self.forward(user, neg_item)
            black_list_loss = 0

        # # traditional mul
        # pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        # neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        # pos score (mlp method)
        pos_features = torch.cat((user_e, pos_item_e), 1)
        pos_features = self.relu(self.bn1(self.fc1(pos_features)))
        pos_item_score = self.fc2(pos_features)

        # pos_item_score = self.sigmoid(pos_item_score)

        # neg score (mlp method)
        neg_features = torch.cat((user_e, neg_item_e), 1)
        neg_features = self.relu(self.bn1(self.fc1(neg_features)))
        neg_item_score = self.fc2(neg_features)
        # neg_item_score = self.sigmoid(neg_item_score)

        # user side
        if self.patient_module:
            pos_user_score = self.UserSide_model.forward(user, pos_item)
            neg_user_score = self.UserSide_model.forward(user, neg_item)
            pos_item_score = self.sigmoid(torch.squeeze(pos_item_score, 1))
            neg_item_score = self.sigmoid(torch.squeeze(neg_item_score, 1))
            predict = torch.cat((pos_item_score + self.user_side_weight * pos_user_score,
                                 neg_item_score + self.user_side_weight * neg_user_score))
            # predict = torch.squeeze(predict, 1)
            # print(pos_user_score.shape, predict.shape)
        else:
            predict = torch.cat((pos_item_score, neg_item_score))
            predict = torch.squeeze(predict, 1)

        # make up
        # simple sum
        # pos_score = pos_item_score + pos_user_score
        # neg_score = neg_item_score + neg_user_score
        # sum after sigmoid
        # pos_score = self.sigmoid(self.sigmoid(pos_item_score) + self.sigmoid(pos_user_score))
        # neg_score = self.sigmoid(self.sigmoid(neg_item_score) + self.sigmoid(neg_user_score))

        target = torch.zeros(len(user) * 2, dtype=torch.float32).to(self.device)
        target[: len(user)] = 1
        rec_loss = self.bce_loss(predict, target)

        l2_loss = self.l2_loss(user_e, pos_item_e, neg_item_e)
        loss = rec_loss + self.reg_weight * l2_loss
        # print('loss: ', loss)
        loss += black_list_loss * self.black_list_weight

        # print('blloss: ', black_list_loss)
        # print(loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        # item side
        if self.black_list_using:
            user_e, item_e, item_bl_e = self.forward(user, item)
        else:
            user_e, item_e = self.forward(user, item)

        features = torch.cat((user_e, item_e), 1)
        features = self.relu(self.bn1(self.fc1(features)))
        score_item = self.fc2(features)

        # make up
        score = score_item

        # return torch.mul(user_e, item_e).sum(dim=1)
        return score

    def full_sort_predict(self, interaction):
        user_index = interaction[self.USER_ID]
        item_index = torch.tensor(range(self.n_items)).to(self.device)

        user = torch.unsqueeze(user_index, dim=1).repeat(1, item_index.shape[0])
        user = torch.flatten(user)
        item = torch.unsqueeze(item_index, dim=0).repeat(user_index.shape[0], 1)
        item = torch.flatten(item)

        # item side
        if self.black_list_using:
            user_e, item_e, item_bl_e = self.forward(user, item)
        else:
            user_e, item_e = self.forward(user, item)
        # mlp method
        features = torch.cat((user_e, item_e), 1)
        features = self.relu(self.bn1(self.fc1(features)))
        score_item = self.fc2(features)
        # score_item = torch.mul(user_e, item_e).sum(dim=1)

        score = score_item

        return score.view(-1)
