import pandas as pd
import torch
import random
import operator
import collections
import numpy as np
import math
import re
import jieba
from collections import Counter
from .metric_evaluation import eval_MRR, eval_NDCG, eval_MAP, eval_recall, eval_precision
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from torch import nn
from BalanRec.utils import (
    set_color
)

class ReRankExposure(nn.Module):
    def __init__(self, config, train_data, test_data, k_list):
        super(ReRankExposure, self).__init__()
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.config = config
        self.n_user = train_data.user_num
        self.n_item = train_data.item_num
        self.dataset_path = config.final_config_dict['data_path']
        self.field2token_id = train_data.field2token_id
        self.inter_feat_train = self.get_feat(train_data.inter_feat)
        self.inter_feat_test = self.get_feat(test_data.inter_feat)
        self.user_feat = self.get_feat(train_data.user_feat)
        self.item_feat = self.get_feat(train_data.item_feat)
        self.k_list = k_list
        self.item_rank = {}
        self.trues = self.get_truth(self.inter_feat_train, self.inter_feat_test)
        self.En, self.Ex, self.Ed = self.init_Evalue('log')

        self.query_dict, self.corpus = self.get_corpus()
        self.item2users = collections.defaultdict(list)
        self.user2ts, self.user2cs = self.patient_sensitivity(list(set(self.inter_feat_test[self.USER_ID].tolist())))
        self.item2DRS = self.cal_DRS()
        self.item2DCF = self.load_DCF()

        self.sigmoid = nn.Sigmoid()
        self.m_value = 3

    def get_feat(self, data):
        # feat ---> dataframe
        columns = data.columns
        return pd.DataFrame({
            col: data[col].tolist() for col in columns
        })

    def init_Evalue(self, n_parameter):
        # n_p ensure
        n_p_core = {}
        if n_parameter == 'log':
            n_p_core = {k: math.log(k, 10) for k in self.k_list}
        elif n_parameter == 'log+ln':
            n_p_core = {k: (math.log(k, 10) + math.log(k)) / 2 for k in self.k_list}
        elif n_parameter == 'ln':
            n_p_core = {k: math.log(k) for k in self.k_list}
        elif n_parameter == 'ln+sqrt':
            n_p_core = {k: (math.log(k) + math.sqrt(k)) / 2 for k in self.k_list}
        elif n_parameter == 'sqrt':
            n_p_core = {k: math.sqrt(k) for k in self.k_list}
        else:
            raise ValueError

        items = list(set(self.item_feat[self.ITEM_ID]))
        users = list(set(self.inter_feat_test[self.USER_ID]))
        users_in_allowed_time = []
        for user in users:
            query_time = self.user_feat[self.user_feat[self.USER_ID] == user]['query_time'].values[0]
            if 1668614400 >= query_time >= 1668614400 - 7776000:  # latest 3 months
                users_in_allowed_time.append(user)

        Ex = {}
        for k in self.k_list:
            sub_k = {}
            for item in items:
                users_of_item = list(set(self.inter_feat_test[self.inter_feat_test[self.ITEM_ID] == item][self.USER_ID]))
                # users_of_item = [elem for elem in users_of_item if elem in users_in_allowed_time]
                sub_k[item] = math.ceil(k * len(users) * len(users_of_item) / len(users_in_allowed_time))
            Ex[k] = sub_k
        self.Ex = Ex
        # print(len(self.inter_feat_test), math.log(len(self.inter_feat_test), k))
        self.En = {k: {item: int(n_p_core[k] * len(self.inter_feat_test) / len(self.item_feat))
                       for item in items} for k in self.k_list}
        self.Ed = {item: 0 for item in range(len(self.item_feat))}

        # re seiri
        for k in self.k_list:
            for item in items:
                if item == 0:
                    continue
                if self.En[k][item] >= self.Ex[k][item]:
                    self.Ex[k][item] = self.En[k][item]

        return self.En, self.Ex, self.Ed

    def get_DP_scores(self, model):
        # get full scores of users and items
        full_scores = torch.FloatTensor([]).to(self.config["device"])
        full_predict_batch_size = 10
        hari = 0
        # embeddings_full
        full_users = torch.IntTensor(
            list(map(int, self.user_feat[self.config.final_config_dict['USER_ID_FIELD']].values[1:])))
        while hari < self.n_user:
            users_part = full_users[hari:hari + full_predict_batch_size]
            hari += full_predict_batch_size
            sub_inter = {self.config.final_config_dict['USER_ID_FIELD']: users_part.to(self.config["device"])}
            with torch.no_grad():
                full_scores = torch.cat((full_scores,  # torch.FloatTensor(sub_tt)).to(self.config["device"]), dim=0)
                                         self.sigmoid(model.full_sort_predict(sub_inter)
                                                      .to(self.config["device"])
                                                      .reshape(len(users_part), self.n_item))), dim=0)
            # print(round(hari / self.n_user, 2))
        return dict(zip(full_users.tolist(), full_scores))

    def original_Evalue(self, item_rank, k_list):
        final_result = {}
        for k in k_list:
            result = {i: 0 for i in range(len(self.item_feat))}
            for user in item_rank.keys():
                sub = item_rank[user][:k]
                for elem in sub:
                    result[elem] += 1
            final_result[k] = result

        return final_result

    def reranking(self, logger, model):
        user_scores = self.get_DP_scores(model)
        users = list(set(self.inter_feat_test[self.USER_ID].tolist()))
        # user_scores = {}
        # for elem in users:
        #     user_scores[elem] = [random.random() for i in range(self.n_item)]

        logger.info(set_color(f"original evaluation: ", "red"))
        item_rank_full = {user: sorted([(i, user_scores[user][i]) for i in range(len(user_scores[user]))],
                                       key=operator.itemgetter(1), reverse=True) for user in users}
        self.item_rank = {user: [elem[0] for elem in item_rank_full[user]] for user in users}
        self.eval_metric_origin(logger, self.k_list, self.item_rank)

        # ramdom pick evaluation
        logger.info(set_color(f"random evaluation: ", "red"))
        scores_dict_random_kumi = {
            key: np.random.permutation(
                [(i, user_scores[key][i].cpu()) for i in range(len(user_scores[key]))]).tolist() for key in
            users}
        scores_dict_random = {key: [int(elem[0]) for elem in scores_dict_random_kumi[key]]
                              for key in users}
        self.eval_metric_origin(logger, self.k_list, scores_dict_random)

        final_recommend_list = dict()
        logger.info(set_color(f"rerank evaluation: ", "red"))
        for k in self.k_list:
            En_parameter = ['ln']  # ['log', 'log+ln', 'ln', 'ln+sqrt', 'sqrt']
            for n_p in En_parameter:
                self.init_Evalue(n_p)
                self.item2users = collections.defaultdict(list)
                self.m_value = [int(2*k)]  # [int(k/2), int(k), int(3*k/2), int(2*k), int(5*k/2), int(3*k)]
                for m in self.m_value:
                    logger.info(set_color(f"m value is {m}; En is {n_p}", "red"))
                    sub_dict = dict()
                    for user in users:
                        score_of_user_sorted = item_rank_full[user]
                        item_score_candidate = []
                        for i in range(len(score_of_user_sorted)):
                            sub_item = score_of_user_sorted[i][0]
                            if self.Ex[k][sub_item] - self.Ed[sub_item] > 0:
                                item_score_candidate.append(score_of_user_sorted[i])
                            if len(item_score_candidate) >= k + m:
                                break

                        step = 0
                        suisen_hit = [0] * (k + m)
                        core_length = min(len(suisen_hit), len(item_score_candidate))
                        w_list = [max(self.En[k][item_score_candidate[i][0]] - self.Ed[item_score_candidate[i][0]], 0)
                                  + k + m - i for i in range(core_length)]
                        while step < k:
                            P_range = [w_list[i] if suisen_hit[i] == 0 else 0 for i in range(core_length)]
                            Q_range = [sum(P_range[:i+1]) if suisen_hit[i] == 0 else 0 for i in range(core_length)]
                            core_random = random.randint(0, sum(P_range))
                            for i in range(len(w_list)):
                                if core_random <= Q_range[i]:
                                    suisen_hit[i] = 1
                                    self.Ed[item_score_candidate[i][0]] += 1
                                    self.item2users[item_score_candidate[i][0]].append(user)
                                    break
                            step += 1
                        sub_dict[user] = [item_score_candidate[i][0] for i in range(core_length) if suisen_hit[i] == 1]

                    self.eval_metric(logger, [k], sub_dict)

            final_recommend_list[k] = sub_dict

        return self.item_rank, final_recommend_list

    def eval_metric_origin(self, logger, k_list, recommend_dict):
        users = list(set(self.inter_feat_test[self.USER_ID].tolist()))
        # evaluation
        Ed_kdict = self.original_Evalue(self.item_rank, k_list)
        trues = self.trues
        precision = eval_precision(k_list, users, recommend_dict, trues)
        recall = eval_recall(k_list, users, recommend_dict, trues)
        MAP = eval_MAP(k_list, users, recommend_dict, trues)
        MRR = eval_MRR(k_list, users, recommend_dict, trues)
        NDCG = eval_NDCG(k_list, users, recommend_dict, trues)
        H_value = []
        Z_value = []
        L_value = []
        I_value = []
        O_value = []
        coverage_value = []
        for k in k_list:
            self.Ed = Ed_kdict[k]
            H_value.append(self.eval_H([k], users, recommend_dict, trues)[0])
            Z_value.append(self.eval_Z([k], users, recommend_dict, trues)[0])
            L_value.append(0)
            I_value.append(self.eval_I([k], users, recommend_dict, trues)[0])
            O_value.append(self.eval_O([k], users, recommend_dict, trues)[0])
            coverage_value.append(self.eval_coverage([k], users, recommend_dict, trues)[0])

        evaluations = {
            'precision': precision,
            'recall': recall,
            'MAP': MAP,
            'MRR': MRR,
            'NDCG': NDCG,
            'H': H_value,
            'Z': Z_value,
            'L': L_value,
            'I': I_value,
            'O': O_value,
            'coverage': coverage_value
        }

        self.evaluation_logger(logger, evaluations, k_list)

        return evaluations

    def eval_metric(self, logger, k_list, recommend_dict):
        users = list(set(self.inter_feat_test[self.USER_ID].tolist()))
        # evaluation
        trues = self.trues
        precision = eval_precision(k_list, users, recommend_dict, trues)
        recall = eval_recall(k_list, users, recommend_dict, trues)
        MAP = eval_MAP(k_list, users, recommend_dict, trues)
        MRR = eval_MRR(k_list, users, recommend_dict, trues)
        NDCG = eval_NDCG(k_list, users, recommend_dict, trues)
        H_value = self.eval_H(k_list, users, recommend_dict, trues)
        Z_value = self.eval_Z(k_list, users, recommend_dict, trues)
        L_value = self.eval_L(k_list, users, recommend_dict, trues)
        I_value = self.eval_I(k_list, users, recommend_dict, trues)
        O_value = self.eval_O(k_list, users, recommend_dict, trues)
        coverage_value = self.eval_coverage(k_list, users, recommend_dict, trues)

        evaluations = {
            'precision': precision,
            'recall': recall,
            'MAP': MAP,
            'MRR': MRR,
            'NDCG': NDCG,
            'H': H_value,
            'Z': Z_value,
            'L': L_value,
            'I': I_value,
            'O': O_value,
            'coverage': coverage_value

        }

        self.evaluation_logger(logger, evaluations, k_list)

        return evaluations

    def get_truth(self, inter_train, inter_test):
        user_id = self.USER_ID
        item_id = self.ITEM_ID

        truth = dict()
        # user2doctors
        user2doctors = pd.read_csv(self.dataset_path + '/user_id_closed_doctors.txt', sep='\t')
        for i in range(len(user2doctors)):
            user = self.field2token_id[user_id][str(user2doctors.loc[i, 'user_id'])]
            docts = [self.field2token_id[item_id][str(elem)] for elem in eval(user2doctors.loc[i, 'doctor_closed'])]
            truth[user] = docts

        # users = list(set(self.user_feat[user_id].values))
        # truth = dict()
        # for user in users:
        #     truth[user] = list(set(inter_test[inter_test[user_id] == user][item_id].values))
        #     truth[user].extend(list(set(inter_train[inter_train[user_id] == user][item_id].values)))
        #     truth[user] = list(set(truth[user]))

        return truth

    def evaluation_logger(self, logger, evaluation, k_list):
        # evaluation logger output
        for i in range(len(k_list)):
            metric_str = f""
            for key in evaluation.keys():
                metric_str += key + " : " + str(evaluation[key][i]) + " "

            logger.info(set_color(f"top{k_list[i]} valid result: ", "blue") + metric_str)

    def eval_H(self, k_list, user_list, items_rank, test_record):
        x_related = {k: sum([1 if self.Ed[key] >= self.Ex[k][key] else 0 for key in self.Ed.keys()]) for k in k_list}
        n_related = {k: sum([1 if self.Ed[key] >= self.En[k][key] else 0 for key in self.Ed.keys()]) for k in k_list}
        return [round((n_related[k] - x_related[k]) / len(self.Ed.keys()), 3) for k in k_list]

    def eval_Z(self, k_list, user_list, items_rank, test_record):
        q_length = len(user_list)
        d_length = len(self.Ed.keys())
        return [round(-sum([self.Ed[key] / (k * q_length) * math.log((self.Ed[key] + 1) / (k * q_length), d_length)
                for key in self.Ed.keys()]), 3) for k in k_list]

    def eval_L(self, k_list, user_list, items_rank, test_record):
        origin_Ed = self.original_Evalue(self.item_rank, self.k_list)
        return [round((1/len(self.Ed.keys())) * sum([max((origin_Ed[k][key] - self.Ed[key]) / (origin_Ed[k][key] + 1), 0)
                                                     for key in self.Ed.keys()]), 3) for k in k_list]

    def eval_I(self, k_list, user_list, items_rank, test_record):
        I_kumi = []
        for user in user_list:
            if self.user2ts[user] == 0:
                continue
            candidate_list = items_rank[user][:k_list[0]]
            box_test = [self.item2DRS[elem] if elem in self.item2DRS.keys() else 0 for elem in candidate_list]
            if len(box_test) == 0:
                DRS_test = 0
            else:
                DRS_test = np.mean(box_test)

            box_true = [self.item2DRS[elem] for elem in test_record[user]]
            if len(box_true) == 0:
                DRS_true = 0
            else:
                DRS_true = np.mean(box_true)

            if DRS_true == 0:
                I_kumi.append(0)
            else:
                I_kumi.append(self.user2ts[user] * (DRS_test - DRS_true) / DRS_true)
        # print(I_kumi)
        return [round(sum(I_kumi) / len(I_kumi), 3)]

    def eval_O(self, k_list, user_list, items_rank, test_record):
        # DCF dict
        O_kumi = []
        for user in user_list:
            if self.user2cs[user] == 0:
                continue
            candidate_list = items_rank[user][:k_list[0]]
            box_test = [self.item2DCF[elem] if elem in self.item2DCF.keys() else 0 for elem in candidate_list]
            if len(box_test) == 0:
                DCF_test = 0
            else:
                DCF_test = np.mean(box_test)

            box_true = [self.item2DCF[elem] for elem in test_record[user]]
            if len(box_true) == 0:
                DCF_true = 0
            else:
                DCF_true = np.mean(box_true)

            if DCF_true == 0:
                O_kumi.append(0)
            else:
                O_kumi.append(self.user2cs[user] * (DCF_true - DCF_test) / DCF_true)
        # print(O_kumi)
        return [round(sum(O_kumi) / len(O_kumi), 3)]

    def eval_coverage(self, k_list, user_list, items_rank, test_record):
        coverage = []
        for k in k_list:
            items_suisen = []
            for user in user_list:
                items_suisen.extend(items_rank[user][:k])
            coverage.append(round(len(set(items_suisen)) / (len(self.item_feat) - 1), 3))
        return coverage

    def load_DCF(self):
        # doctor fame
        columns = list(self.item_feat.columns)
        unused = [self.ITEM_ID, 'department', 'doctor_title', 'education_title', 'consultation_amount',
                  'active_years',
                  'patient_recommendation_score']
        [columns.remove(elem) for elem in unused]

        item_DOOF = dict(zip(self.item_feat[self.ITEM_ID].values, self.item_feat[columns].values))

        DOOF = []
        items = list(set(self.item_feat[self.ITEM_ID]))
        for item in items:
            DOOF.append(item_DOOF[item])

        # standard
        scaler = MinMaxScaler()
        values = scaler.fit_transform(DOOF).tolist()
        final_item_DOOF = {items[i]: round(sum(values[i]) / len(values[i]), 4) for i in range(len(items))}

        return final_item_DOOF

    def cal_DRS(self):
        users = list(set(list(self.inter_feat_train[self.USER_ID])))
        user_feat_index = []
        for user in users:
            user_feat_index.extend(self.user_feat[self.user_feat[self.USER_ID] == user].index)
        user_feat = self.user_feat.loc[user_feat_index, :]

        item_DRS = dict()
        max_rt = max(list(user_feat['response_time'].values))
        for item in list(set(self.item_feat[self.ITEM_ID])):
            sub_users = list(
                set(self.inter_feat_train[self.inter_feat_train[self.ITEM_ID] == item][self.USER_ID].values))
            if len(sub_users) == 0:
                item_DRS[item] = 0
            else:
                DRS_box = []
                for user in sub_users:
                    sub_users_feat = user_feat[user_feat[self.USER_ID] == user]
                    rt = - (sub_users_feat['response_time'].values[0] / max_rt) + 1
                    DRS_box.append(rt)
                item_DRS[item] = round(sum(DRS_box) / len(DRS_box), 4)

        return item_DRS

    def get_corpus(self):
        query_file = pd.read_csv(self.dataset_path + '/LDA_related/user_text.csv', sep='\t')
        u2query = dict(zip(query_file[self.USER_ID].values, query_file['query'].values))
        users_origin = [elem for elem in query_file[self.USER_ID].values
                        if str(elem) in self.field2token_id[self.USER_ID].keys()]
        users_all = [self.field2token_id[self.USER_ID][str(elem)] for elem in users_origin]
        queries = [u2query[elem] for elem in users_origin]
        query_dict = dict(zip(users_all, queries))

        corpus = []
        for user in list(set(self.inter_feat_train[self.USER_ID])):
            if user in [0, '0']:
                continue
            words_cut = self.seg_depart(query_dict[user])
            corpus.append([wrd for wrd in words_cut if len(wrd) > 0 and wrd != ' '])

        return [query_dict, corpus]

    def patient_sensitivity(self, users):
        # time and cost sensitive of patients
        # time and cost words
        time_wrds = ['现在', '严重', '时间', '加重', '多久', '什么时候', '多长时间', '当天', '早点', '需要多久', '情况严重', '很快',
                     '急需', '急', '快']
        cost_wrds = ['费用', '便宜', '昂贵', '贵', '省钱', '多少钱']
        final_ts = dict()
        final_cs = dict()
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
            final_ts[user] = time_tfidf_sum
            final_cs[user] = cost_tfidf_sum

        return final_ts, final_cs

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
