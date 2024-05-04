import pandas as pd
import torch
import math
import jieba
import re
import numpy as np
from tqdm import tqdm
from torch import nn
from BalanRec.model.init import xavier_normal_initialization
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BalanRecUserSide(nn.Module):
    def __init__(self, config, dataset):
        super(BalanRecUserSide, self).__init__()
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.dialog_to_use_num = config['dialog_to_use_num']
        self.QUERY_ID = config['QUERY_FIELD']
        self.txt_embeddings = dataset.txt_embeddings
        self.inter_feat = self.get_feat(dataset.inter_feat)
        self.user_feat = self.get_feat(dataset.user_feat)
        self.item_feat = self.get_feat(dataset.item_feat)
        self.device = config["device"]
        self.dataset_path = config.final_config_dict['data_path']
        self.field2token = dataset.field2token_id

        # attributes calculate
        self.query_dict, self.corpus = self.get_corpus()
        self.item_DRS, self.item_DOOF = self.item_new_feat_standard()
        self.user_ts, self.user_cs = self.user_new_feat_standard()

        # MLP layer
        self.in_size = np.array(list(self.item_DOOF.values())).shape[1]
        self.hid_size = 4
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=self.hid_size)
        self.fc1 = nn.Linear(self.in_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # parameters initialization
        self.apply(xavier_normal_initialization)

        # the weight parameter of loss_time and loss_fame
        self.weight = 1.0
        # self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def get_feat(self, data):
        # feat ---> dataframe
        columns = data.columns
        return pd.DataFrame({
            col: data[col].tolist() for col in columns
        })

    def forward(self, user, item):
        # user and item matching score from the side of user
        kumi = [(user.tolist()[i], item.tolist()[i]) for i in range(len(user))]

        # score got from mlp of every pair of user and item
        loss_time = []
        ts_of_users = []
        cs_of_users = []
        DRS_of_items = []
        DOOF_of_items = []
        for u_s, i_s in kumi:
            ts_of_users.append(self.user_ts[u_s])
            DRS_of_items.append(self.item_DRS[i_s])
            # loss_time.append(self.user_ts[u_s] * (1 - self.item_DRS[i_s]))
            cs_of_users.append(self.user_cs[u_s])
            DOOF_of_items.append(sum(self.item_DOOF[i_s]) / len(self.item_DOOF[i_s]))

        # DOOF mlp module
        score = torch.FloatTensor(DOOF_of_items).to(self.device)  # features
        # features = self.relu(self.bn1(self.fc1(features)))
        # score = self.fc2(features)
        # score = self.sigmoid(score)
        score_fame = torch.mul(self.sigmoid(torch.FloatTensor(cs_of_users).to(self.device)), score.squeeze())
        score_time = torch.mul(self.sigmoid(torch.FloatTensor(ts_of_users).to(self.device)),
                               self.sigmoid(torch.FloatTensor(DRS_of_items).to(self.device)))

        return self.weight * score_time + score_fame

    def item_new_feat_standard(self):
        item_DRS = self.get_DRS()
        item_DOOF, columns = self.get_DOOF()
        DRS = []
        DOOF = []
        items = list(set(self.item_feat[self.ITEM_ID]))
        for item in items:
            DRS.append(item_DRS[item])
            DOOF.append(item_DOOF[item])

        # DOOF = list(map(list, zip(*DOOF)))
        # new_feat = [DRS] + DOOF
        # new_feat = np.stack(new_feat, axis=1)

        # standard
        scaler = MinMaxScaler()
        values = scaler.fit_transform(DOOF)
        # final_item_DRS = dict(zip(items, values[:, 0]))
        # final_item_DOOF = dict(zip(items, values[:, 1:]))
        final_item_DRS = dict(zip(items, DRS))
        final_item_DOOF = dict(zip(items, values.tolist()))

        return final_item_DRS, final_item_DOOF

    def get_DRS(self):
        # doctor response time related
        # start and end of query time
        users = list(set(list(self.inter_feat[self.USER_ID])))
        user_feat_index = []
        for user in users:
            user_feat_index.extend(self.user_feat[self.user_feat[self.USER_ID] == user].index)
        user_feat = self.user_feat.loc[user_feat_index, :]
        # query_time = sorted(list(set(list(user_feat['query_time'].values))))
        # q2order = dict(zip(query_time, range(1, len(query_time) + 1, 1)))
        # r_s_time = sorted(list(set(list(user_feat['first_response_time'] - user_feat['query_time']))))
        # r_s2order = dict(zip(r_s_time, range(len(r_s_time))))
        #
        # item_DRS = dict()
        # for item in list(set(self.item_feat[self.ITEM_ID])):
        #     sub_users = list(
        #         set(self.inter_feat[self.inter_feat[self.ITEM_ID] == item][self.USER_ID].values))
        #     if len(sub_users) == 0:
        #         item_DRS[item] = 1
        #     else:
        #         DRS_box = []
        #         for user in sub_users:
        #             sub_users_feat = user_feat[user_feat[self.USER_ID] == user]
        #             r_s_user = r_s2order[
        #                 (sub_users_feat['first_response_time'] - sub_users_feat['query_time']).values[0]]
        #             q_user = q2order[sub_users_feat['query_time'].values[0]]
        #             DRS_box.append(math.exp(-(r_s_user / q_user)))
        #         DRS_box = [elem for elem in DRS_box if elem != 0]
        #         item_DRS[item] = round(min(DRS_box), 3)

        item_DRS = dict()
        max_rt = max(list(user_feat['response_time'].values))
        for item in list(set(self.item_feat[self.ITEM_ID])):
            sub_users = list(
                set(self.inter_feat[self.inter_feat[self.ITEM_ID] == item][self.USER_ID].values))
            if len(sub_users) == 0:
                item_DRS[item] = 0
            else:
                DRS_box = []
                for user in sub_users:
                    sub_users_feat = user_feat[user_feat[self.USER_ID] == user]
                    rt = - (sub_users_feat['response_time'].values[0] / max_rt) + 1
                    DRS_box.append(rt)
                item_DRS[item] = round(sum(DRS_box) / len(DRS_box), 3)

        return item_DRS

    def get_DOOF(self):
        # doctor fame
        columns = list(self.item_feat.columns)
        unused = [self.ITEM_ID, 'department', 'doctor_title', 'education_title', 'consultation_amount', 'active_years',
                  'patient_recommendation_score']
        [columns.remove(elem) for elem in unused]

        return dict(zip(self.item_feat[self.ITEM_ID].values, self.item_feat[columns].values)), columns

    def user_new_feat_standard(self):
        users = list(set(self.user_feat[self.USER_ID].values))
        users.remove(0)
        ts_cs = self.get_time_cost_sensitive(users)
        ts = []
        cs = []
        for user in users:
            ts.append(ts_cs[user][0])
            cs.append(ts_cs[user][1])

        # # standard
        # scaler = MinMaxScaler()
        # values = scaler.fit_transform(np.stack([ts] + [cs]))
        # user_ts = dict(zip(users, values[0, :]))
        # user_cs = dict(zip(users, values[1, :]))

        user_ts = dict(zip(users, ts))
        user_cs = dict(zip(users, cs))

        return user_ts, user_cs

    def get_corpus(self):
        query_file = pd.read_csv(self.dataset_path + '/LDA_related/user_text.csv', sep='\t')
        u2query = dict(zip(query_file[self.USER_ID].values, query_file['query'].values))
        users_origin = [elem for elem in query_file[self.USER_ID].values
                        if str(elem) in self.field2token[self.USER_ID].keys()]
        users_all = [self.field2token[self.USER_ID][str(elem)] for elem in users_origin]
        queries = [u2query[elem] for elem in users_origin]
        query_dict = dict(zip(users_all, queries))

        corpus = []
        for user in list(set(self.inter_feat[self.USER_ID])):
            if user in [0, '0']:
                continue
            words_cut = self.seg_depart(query_dict[user])
            corpus.append([wrd for wrd in words_cut if len(wrd) > 0 and wrd != ' '])

        return [query_dict, corpus]

    def get_time_cost_sensitive(self, users):
        # time and cost sensitive of patients

        # # whole condition of words times
        # all_query = ' '.join(query_dict.values())
        # words_cut = self.seg_depart(all_query)
        # counts = dict()
        # for word in words_cut:
        #     if len(word) < 1:
        #         continue
        #     else:
        #         counts[word] = counts.get(word, 0) + 1
        # items = list(counts.items())
        # items.sort(key=lambda x: x[1], reverse=True)
        # words_times = pd.DataFrame({
        #     'word': [elem[0] for elem in items],
        #     'times': [elem[1] for elem in items]
        # })
        # words_times.to_csv(self.dataset_path + '/LDA_related/word_times.txt', sep='\t', index=False)

        # time and cost words
        time_wrds = ['急需', '急', '情况严重', '早点', '快', '很快', '严重', '加重', '需要多久', '多久', '什么时候',
                     '多长时间', '现在', '当天']
        cost_wrds = ['多少钱', '费用', '昂贵', '贵', '便宜', '省钱']
        final_ts_cs = dict()
        n_containing = dict()
        for wrd in time_wrds + cost_wrds:
            n_containing[wrd] = sum([1 for elem in self.corpus if wrd in elem])

        for user in users:
            query = self.query_dict[user]
            words_cut = self.seg_depart(query)
            words_cut = [elem for elem in words_cut if elem not in ['', ' ']]
            count_dict = Counter(words_cut)

            # time
            time_tfidf = []
            for wrd in time_wrds:
                tf = count_dict[wrd] / len(words_cut)
                idf = math.log(len(self.corpus) / (1 + n_containing[wrd]))
                time_tfidf.append(tf * idf)
            time_tfidf_sum = sum(time_tfidf)

            # cost
            cost_tfidf = []
            for wrd in cost_wrds:
                tf = count_dict[wrd] / len(words_cut)
                idf = math.log(len(self.corpus) / (1 + n_containing[wrd]))
                cost_tfidf.append(tf * idf)
            cost_tfidf_sum = sum(cost_tfidf)
            final_ts_cs[user] = [time_tfidf_sum, cost_tfidf_sum]

        return final_ts_cs

    def stopwordslist(self):
        # create stop_words list
        path = self.dataset_path + '/LDA_related/Chinese_stop_words.txt'
        stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
        return stopwords

    def seg_depart(self, sentence):
        # chinese split for sentence and stop words
        dict_path = self.dataset_path + '/LDA_related/userdict.txt'
        jieba.load_userdict(dict_path)
        pattern = "[_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/\"]"
        sentence = re.sub(pattern, '', sentence)
        sentence_depart = jieba.cut(sentence.strip())
        stopwords = self.stopwordslist()
        # output result
        outstr = []
        # stop words
        for word in sentence_depart:
            if word not in stopwords:
                if word != '\t':
                    outstr.append(word)
        return outstr
