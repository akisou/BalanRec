import collections
import re
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import jieba
import json
import torch
import operator
from collections import Counter
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
#pd.set_option('display.max_colwidth', 200)

def data_extract_kg():
    # extract the entity of kg from raw data
    root_path = "E:/program/PycharmProjects/doctor_recomend/data/haodf/"
    save_path = '../../dataset/DP_chaos/'
    department = ['心血管内科', '神经内科', '消化内科', '内分泌科', '呼吸内科', '感染内科', '眼科综合',
                  '口腔科综合', '肿瘤内科', '妇科', '男科', '中医综合', '儿科综合', '康复科', '精神科']
    attrs = ['doctor_id', 'doctor_title', 'education_title', 'hospital', 'consultation_amount',
             'patient_recommendation_score', 'further_experience', 'profile', 'social_title', 'work_experience',
             'education_experience', 'total_access', 'total_article', 'total_patient', 'total_evaluation',
             'thanks_letter_num', 'thanks_present', 'new_in_time',
             'cure_experience', 'evaluation_type_set', 'cure_satisfaction', 'attitude_satisfaction']

    r_source = []
    r_target = []
    r_relation = []

    k_source = []
    k_target = []
    k_relation = []

    doctor_info = pd.DataFrame(columns=['doctor_id', 'department', 'doctor_title', 'education_title',
                                        'consultation_amount', 'patient_recommendation_score', 'profile', 'total_access',
                                        'total_article', 'total_patient', 'total_evaluation', 'thanks_letter_num',
                                        'thanks_present', 'active_years', 'cure_satisfaction', 'attitude_satisfaction'])

    for depart in department:
        doctor = pd.read_csv(root_path + depart + '/doctor.csv')
        for i in range(len(doctor)):
            did = doctor.loc[i, 'doctor_id']
            patients_of_doctor_path = root_path + depart + '/' + str(did)
            if (not os.path.exists(patients_of_doctor_path)) or (len(os.listdir(patients_of_doctor_path)) == 0):
                continue

            doctor_info_single = []
            doctor_info_single.append(did)
            doctor_info_single.append(depart)
            for attr in attrs[1:]:
                if attr in ['doctor_title', 'education_title', 'consultation_amount', 'patient_recommendation_score',
                            'profile', 'total_access', 'total_article', 'total_patient', 'total_evaluation',
                            'thanks_letter_num', 'thanks_present', 'cure_satisfaction', 'attitude_satisfaction']:
                    if pd.isnull(doctor.loc[i, attr]):
                        doctor_info_single.append("")
                    else:
                        doctor_info_single.append(doctor.loc[i, attr])
                elif attr == 'new_in_time':
                    start_year = int(doctor.loc[i, attr][:4])
                    doctor_info_single.append(2022 - start_year)
                elif attr == 'hospital':
                    for hos in eval(doctor.loc[i, attr]):
                        if hos != None:
                            r_source.append(did)
                            posi = hos.find('医院')
                            r_target.append(hos[:posi + 2])
                            r_relation.append('doctor.work_in.hospital')
                elif attr in ['further_experience', 'work_experience', 'education_experience']:
                    if not pd.isnull(doctor.loc[i, attr]):
                        for exper in eval(doctor.loc[i, attr]):
                            r_source.append(did)
                            institute = exper.split(';')[1]
                            posi = max(institute.find('医院'), institute.find('大学'))
                            r_target.append(institute[:posi + 2])
                            if attr in ['further_experience', 'work_experience']:
                                r_relation.append('doctor.work_in.hospital')
                            else:
                                r_relation.append('doctor.study_in.university')
                elif attr in ['cure_experience', 'evaluation_type_set']:
                    for label in eval(doctor.loc[i, attr])[0]:
                        k_source.append(did)
                        k_target.append(label)
                        k_relation.append('doctor.cure_disease.disease')
                elif attr == 'social_title':
                    special = ['中国医师协会', '中华医学会', '中国医生协会', '中国医学会']
                    bad_words = ['担任', '现任', '兼任']
                    if pd.isnull(doctor.loc[i, attr]):
                        continue
                    for kumi in eval(doctor.loc[i, attr]):
                        for elem in re.split(r"[.,。，]", kumi):
                            position = elem.find('会')
                            special_kumi = [elem.find(term) for term in special]
                            if (np.array(special_kumi) < 0).all():
                                if position > 0:
                                    r_source.append(did)
                                    r_target.append(elem[:position + 1])
                                    r_relation.append('doctor.member_of.institute')
                            else:
                                sub_elem = elem[position + 1:]
                                sub_position = sub_elem.find('会')
                                if sub_position > 0:
                                    r_source.append(did)
                                    institute = elem[:position + sub_position + 2]
                                    for word in bad_words:
                                        institute = institute.replace(word, '')
                                    r_target.append(institute)
                                    r_relation.append('doctor.member_of.institute')
                else:
                    print('sth wrong!')

            doctor_info.loc[len(doctor_info.index)] = doctor_info_single

        doctor_rg = pd.DataFrame({
            'doctor_id': r_source,
            'relation': r_relation,
            'target': r_target
        })

        doctor_kg = pd.DataFrame({
            'doctor_id': k_source,
            'relation': k_relation,
            'target': k_target
        })

        doctor_rg = doctor_rg.drop_duplicates(doctor_rg.keys(), keep='first')
        doctor_kg = doctor_kg.drop_duplicates(doctor_kg.keys(), keep='first')

        doctor_rg.sort_values(by='doctor_id', inplace=True, ascending=True)
        doctor_rg = doctor_rg.reset_index(drop=True)

        doctor_kg.sort_values(by='doctor_id', inplace=True, ascending=True)
        doctor_kg = doctor_kg.reset_index(drop=True)

        doctor_rg.to_csv(save_path + 'doctor_rg.txt', sep='\t', index=False)
        doctor_kg.to_csv(save_path + 'doctor_kg.txt', sep='\t', index=False)

        doctor_info.to_csv(save_path + 'doctor_info.txt', sep='\t', index=False)

def text_data_process():
    # extract text data from raw data
    # include profile, query, dialogue
    root_path = "E:/program/PycharmProjects/doctor_recomend/data/haodf/"
    save_path = '../../dataset/DP_chaos/'
    department_cn = ['心血管内科', '神经内科', '消化内科', '内分泌科', '呼吸内科', '感染内科', '眼科综合',
                     '口腔科综合', '肿瘤内科', '妇科', '男科', '中医综合', '儿科综合', '康复科', '精神科']

    result = pd.DataFrame(columns=['p_id', 'd_id', 'department', 'profile', 'query', 'dialogue',
                                   'query_time', 'first_response_time'])

    for i in range(len(department_cn)):
        sub_query = []
        sub_p_id = []
        sub_d_id = []
        sub_profile = []
        sub_dialogue = []
        sub_total_communication = []
        sub_doctor_reply_times = []
        sub_only_robot = []
        sub_wrong_department = []
        sub_has_advice = []
        sub_first_response_time = []
        sub_query_time = []

        sub_path = root_path + department_cn[i]
        doctors = os.listdir(sub_path)
        doctors.remove('doctor.csv')
        for j in tqdm(range(len(doctors))):
            if len(os.listdir(sub_path + '/' + doctors[j])) == 0:
                continue
            doctor = pd.read_csv(sub_path + '/doctor.csv')
            doctor['doctor_id'] = doctor['doctor_id'].astype(str)
            profile = doctor[doctor['doctor_id'] == str(doctors[j])]['profile'].values
            if len(profile) > 0:
                profile = profile[0]
            else:
                continue

            patients = pd.read_csv(sub_path + '/' + doctors[j] + '/' + 'patient.csv')
            patients = patients[['patient_id', 'total_communication_times', 'doctor_reply_times', 'has_advice', 'query',
                                 'disease_label', 'disease_description']]
            reject_word = ['别的科室', '错科室', '科室不对']
            for k in range(len(patients)):
                # dialogue
                dial_whole = pd.read_csv(sub_path + '/' + doctors[j] + '/' + str(patients.loc[k, 'patient_id']) + '.txt'
                                         , sep='\t')
                dial = dial_whole[~dial_whole['speaker'].isin(['小牛医助'])]
                words = dial['word'].values

                # if only robot
                if len(words) == 0:
                    only_robot = 1
                else:
                    only_robot = 0

                dialogue = ''
                for hari in range(len(words)):
                    stop_words = ['仅主诊医生和患者本人可见', '天气逐渐转凉', '您已报到成功', '您好欢迎您使用网上诊室功能']
                    bad_word = 0
                    for stop in stop_words:
                        if str(words[hari]).find(stop) >= 0:
                            bad_word = 1
                    if bad_word == 0:
                        dialogue = dialogue + re.sub(r'\(.+留言\)', "",
                                                     str(words[hari]).replace('\n', ' ').replace('\r', ' ') + ' ')

                # if wrong department
                wrong_department = 0
                for wrd in reject_word:
                    if dialogue.find(wrd) >= 0:
                        wrong_department = 1

                # query time
                query_time_effect = ''
                first_response_time_effect = ''
                disease_description = patients.loc[k, 'disease_description']
                if pd.isnull(disease_description):
                    continue
                if len(disease_description) >= 14:
                    query_time_field = disease_description[-14:]
                    query_time_find = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", query_time_field)
                    if query_time_find:
                        query_time = query_time_find.group(0)
                        query_time_effect = int(time.mktime(datetime.strptime(query_time, '%Y-%m-%d').timetuple()))
                    else:
                        continue
                else:
                    continue

                # first response time of doctors
                dial = dial_whole[dial_whole['speaker'] == '医生']
                firsttime = dial['datetime'].values
                if len(firsttime) > 0:
                    firsttime = str(firsttime[0])
                    # time transform
                    complete_pattern = r"(\d{4}.\d{1,2}.\d{1,2})"
                    part_pattern = r"(\d{1,2}.\d{1,2})"
                    null_pattern = '-1'
                    special_time = {'今天': '2022.11.18', '昨天': '2022.11.17'}
                    if firsttime == null_pattern:
                        firsttime = -1
                    elif re.search(complete_pattern, firsttime):
                        firsttime = firsttime
                    elif re.search(part_pattern, firsttime):
                        firsttime = '2022.' + firsttime
                    else:
                        firsttime = special_time[firsttime]
                    first_response_time_effect = int(time.mktime(datetime.strptime(firsttime, '%Y.%m.%d').timetuple()))
                else:
                    first_response_time_effect = -1

                # print(dialogue)
                sub_total_communication.append(patients.loc[k, 'total_communication_times'])
                sub_doctor_reply_times.append(patients.loc[k, 'doctor_reply_times'])
                sub_only_robot.append(only_robot)
                sub_wrong_department.append(wrong_department)
                sub_has_advice.append(patients.loc[k, 'has_advice'])
                sub_p_id.append(patients.loc[k, 'patient_id'])
                sub_query.append(patients.loc[k, 'query'])
                sub_d_id.append(doctors[j])
                sub_profile.append(profile)
                sub_dialogue.append(dialogue)
                sub_query_time.append(query_time_effect)
                sub_first_response_time.append(first_response_time_effect)

        sub_data = pd.DataFrame({
            'p_id': sub_p_id,
            'd_id': sub_d_id,
            'department': department_cn[i],
            'profile': sub_profile,
            'query': sub_query,
            'dialogue': sub_dialogue,
            'total_communication_times': sub_total_communication,
            'doctor_reply_times': sub_doctor_reply_times,
            'only_robot': sub_only_robot,
            'wrong_department': sub_wrong_department,
            'has_advice': sub_has_advice,
            'query_time': sub_query_time,
            'first_response_time': sub_first_response_time
        })

        # print(sub_data)
        result = pd.concat([result, sub_data], axis=0, ignore_index=True)

    # print(result)
    result.to_csv(save_path + 'text_data.csv', sep='\t', index=False)

def item_process():
    data_path = '../../dataset/DP_chaos/'
    save_path = '../../dataset/OMC-100k-origin/'

    # item part
    items_info = pd.read_csv(data_path + 'doctor_info.txt', sep='\t')
    pid_qualified = pd.read_csv(data_path + 'pid_qualified.txt', sep='\t')
    pids = list(set(pid_qualified['pid'].values))
    text_data = pd.read_csv(data_path + 'text_data_full_info.csv', sep='\t')
    text_data = text_data[text_data['p_id'].isin(pids)]
    items_info = items_info[items_info['item_id'].isin(list(set(text_data['d_id'].values)))]
    # items_info = items_info[~items_info['item_id'].isin([164967473])]
    # items_info['item_id:token'] = items_info['doctor_id']
    items_info = items_info.reset_index(drop=True)
    columns = {
            'item_id':                      'token',
            'department':                   'single_str',
            'hospital':                     'list',
            'doctor_title':                 'single_str',
            'education_title':              'single_str',
            'consultation_amount':          'float',
            'patient_recommendation_score': 'float',
            'total_access':                 'float',
            'total_article':                'float',
            'total_patient':                'float',
            'total_evaluation':             'float',
            'thanks_letter_num':            'float',
            'thanks_present':               'float',
            'active_years':                 'float',
            'cure_satisfaction':            'float',
            'attitude_satisfaction':        'float',
            'comprehensive_rank':           'float',
            'frank':                        'float'
    }
    col_order = ['item_id:token', 'department:single_str', 'doctor_title:single_str',
                 'education_title:single_str', 'consultation_amount:float', 'patient_recommendation_score:float',
                 'total_access:float', 'total_article:float', 'total_patient:float', 'total_evaluation:float',
                 'thanks_letter_num:float', 'thanks_present:float', 'active_years:float', 'cure_satisfaction:float',
                 'attitude_satisfaction:float', 'comprehensive_rank:float', 'frank:float']

    remove_list = []
    for i in range(len(items_info)):
        if pd.isnull(items_info.loc[i, 'profile']):
            remove_list.append(i)
            continue
        for col in columns.keys():
            value = items_info.loc[i, col]
            if type(value) == int or type(value) == float or col in ['department', 'doctor_title', 'education_title', 'hospital']:
                continue
            elif type(value) == str:
                pattern = re.compile(r'(^[-+]?([1-9][0-9]*|0)(\.[0-9]+)?$)')
                ifnumeric = pattern.match(items_info.loc[i, col])
                if not ifnumeric:
                    items_info.loc[i, col] = items_info.loc[i-1, col]

            items_info[str(col)+':'+columns[col]] = items_info[col]

    for col in columns.keys():
        if col in ['department', 'doctor_title', 'education_title', 'hospital']:
            items_info[str(col)+':'+columns[col]] = items_info[col]
        items_info = items_info.drop(col, axis=1)
    items_info = items_info.drop(remove_list, axis=0)
    items_info = items_info.drop(['profile'], axis=1)
    items_info = items_info[col_order]
    items_info.to_csv(save_path+'OMC-100k.item', sep='\t', index=False)

def user_process():
    data_path = '../../dataset/DP_chaos/'
    save_path = '../../dataset/OMC-100k-origin/'

    # user part
    user_columns = {
        'p_id': 'user_id:token',
        'department': 'department:single_str',
        'query_time': 'query_time:float',
        'first_response_time': 'first_response_time:float'
    }

    # pid2point
    pid_qualified = pd.read_csv(data_path + 'pid_qualified.txt', sep='\t')
    pid2point = dict(zip(pid_qualified['pid'], pid_qualified['sensitive_point']))

    text_data = pd.read_csv(data_path + 'sampled/positive_text_data.csv', sep='\t')
    # posi output
    original_cols = text_data.columns
    for col in user_columns.keys():
        text_data[user_columns[col]] = text_data[col]
    user_data = text_data.drop(user_columns.keys() | original_cols, axis=1)


    # nega output
    nega_text_data = pd.read_csv(data_path + 'sampled/negative_text_data.csv', sep='\t')
    original_cols = nega_text_data.columns
    for col in user_columns.keys():
        nega_text_data[user_columns[col]] = nega_text_data[col]
    nega_user_data = nega_text_data.drop(user_columns.keys() | original_cols, axis=1)

    # response time part
    point_set = sorted(list(set([eval(pid_qualified.loc[i, 'sensitive_point'])[0]
                                 for i in range(len(pid_qualified))])), reverse=True)
    p2time = dict(zip(point_set, [elem * 30 for elem in list(range(1, len(point_set) + 1, 1))]))
    posi_time_point = []
    for i in range(len(text_data)):
        pid = text_data.loc[i, 'p_id']
        point = eval(pid2point[pid])[0]
        posi_time_point.append(random.randint(p2time[point] - 30, p2time[point]))
    nega_time_point = []
    for i in range(len(nega_text_data)):
        pid = nega_text_data.loc[i, 'p_id']
        point = eval(pid2point[pid])[0]
        nega_time_point.append(random.randint(p2time[point] - 30, p2time[point]))

    user_data['response_time:float'] = posi_time_point
    nega_user_data['response_time:float'] = nega_time_point

    user_data.to_csv(save_path + 'OMC-100k.user', sep='\t', index=False)
    nega_user_data.to_csv(save_path + 'OMC-100k_nega.user', sep='\t', index=False)

def sample_data():
    data_path = '../../dataset/DP_chaos/'
    save_path = '../../dataset/OMC-100k-origin/'

    doctor_info = pd.read_csv(save_path + 'OMC-100k.item', sep='\t')
    text_data = pd.read_csv(data_path + 'text_data.csv', sep='\t')

    # pid with time or cost sensitivity
    pid_qualified = pd.read_csv('../../dataset/DP_chaos/pid_qualified.txt', sep='\t')
    pid_allowed = list(set(list(pid_qualified['pid'])))

    text_data = text_data[text_data['p_id'].isin(pid_allowed)]
    # text_data = text_data[~text_data['d_id'].isin([164967473])]
    text_data = text_data.reset_index(drop=True)

    patient_num = 1000
    dialogue_length_min = 10
    positive_selected_index = []
    negative_selected_index = []
    original_cols = text_data.columns
    weight = 0.2
    neg_doctor_reply_max = 4
    start_timestamp = 1388505600
    for doct in tqdm(list(set(doctor_info['item_id:token']))):
        sub_data = text_data[text_data['d_id'] == doct]
        sub_data['query_length'] = [len(sub_data.loc[index, 'query']) if type(sub_data.loc[index, 'query']) == str
                                    else 0 for index in sub_data.index]
        sub_data['dialogue_length'] = [len(sub_data.loc[index, 'dialogue'])
                                       if type(sub_data.loc[index, 'dialogue']) == str
                                       else 0 for index in sub_data.index]
        sub_data['time_unequal'] = [1
                                    if (sub_data.loc[index, 'query_time'] != sub_data.loc[index, 'first_response_time'])
                                       and (sub_data.loc[index, 'first_response_time'] > 0) else 0
                                    for index in sub_data.index]
        # positive sample
        posi_sub_data = sub_data[(sub_data['dialogue_length'] >= dialogue_length_min) &
                                 (sub_data['first_response_time'] >= start_timestamp) &
                                 (sub_data['query_time'] >= start_timestamp)]  # & (sub_data['time_unequal'] == 1)]
        posi_sub_data = posi_sub_data.sort_values(by=['query_length', 'dialogue_length'],
                                                  ascending=False)
        posisam = posi_sub_data.index # [:patient_num]
        positive_selected_index.extend(posisam)

        # negative sample
        sub_nage_selected_index = []
        sub_data['doctor_weight'] = [sub_data.loc[i, 'doctor_reply_times'] / sub_data.loc[i, 'total_communication_times']
                                     if sub_data.loc[i, 'total_communication_times'] != 0 else 0
                                     for i in sub_data.index]
        nega_sub_data = sub_data.sort_values(by=['doctor_reply_times'], ascending=True)
        # print(nega_sub_data['doctor_reply_times'])
        nega_sub_data = nega_sub_data[nega_sub_data['doctor_reply_times'] <= neg_doctor_reply_max]
        sub_nage_selected_index.extend(nega_sub_data.index[:int(weight*len(nega_sub_data))])
        if len(sub_nage_selected_index) >= patient_num:
            sub_nage_selected_index = sub_nage_selected_index[:patient_num]
        else:
            nega_sub_data = nega_sub_data.loc[list(set(nega_sub_data.index)-set(sub_nage_selected_index)), :]
            nega_sub_data = nega_sub_data.sort_values(by=['query_length'], ascending=True)
            nega_choice = nega_sub_data[nega_sub_data['query_length'] <= 4]
            sub_nage_selected_index.extend(nega_choice.index[:int(weight*len(nega_sub_data))])
            if len(sub_nage_selected_index) >= patient_num:
                sub_nage_selected_index = sub_nage_selected_index[:patient_num]
            else:
                nega_sub_data = nega_sub_data.loc[list(set(nega_sub_data.index) - set(sub_nage_selected_index)), :]
                nega_choice = nega_sub_data[nega_sub_data['only_robot'] == 1]
                sub_nage_selected_index.extend(nega_choice.index)
                if len(sub_nage_selected_index) >= patient_num:
                    sub_nage_selected_index = sub_nage_selected_index[:patient_num]
                else:
                    nega_sub_data = nega_sub_data.loc[list(set(nega_sub_data.index) - set(sub_nage_selected_index)), :]
                    nega_choice = nega_sub_data[nega_sub_data['wrong_department'] == 1]
                    sub_nage_selected_index.extend(nega_choice.index)
                    if len(sub_nage_selected_index) >= patient_num:
                        sub_nage_selected_index = sub_nage_selected_index[:patient_num]
                    else:
                        nega_sub_data = nega_sub_data.loc[list(set(nega_sub_data.index) - set(sub_nage_selected_index)),
                                        :]
                        nega_choice = nega_sub_data[nega_sub_data['has_advice'] == 0]
                        sub_nage_selected_index.extend(nega_choice.index)
                        if len(sub_nage_selected_index) >= patient_num:
                            sub_nage_selected_index = sub_nage_selected_index[:patient_num]
        negative_selected_index.extend(sub_nage_selected_index)

    positive_text_data = text_data.loc[positive_selected_index, original_cols]
    positive_text_data.to_csv(data_path + 'sampled/positive_text_data_latest.csv', sep='\t', index=False)

    negative_selected_index = [elem for elem in negative_selected_index if elem not in positive_selected_index]
    negative_text_data = text_data.loc[negative_selected_index, original_cols]
    negative_text_data.to_csv(data_path + 'sampled/negative_text_data_latest.csv', sep='\t', index=False)

    origin = pd.read_csv('../../dataset/DP_chaos/text_data.csv', sep='\t')
    origin_statis = origin['d_id'].value_counts()
    origin_statistics = pd.DataFrame({
        'did': origin_statis.index.tolist(),
        'count': origin_statis.values.tolist()
    })
    print(len(origin))
    origin_statistics.to_csv('../../dataset/DP_chaos/text_data_statistics.txt', sep='\t', index=False)

    posi = pd.read_csv('../../dataset/DP_chaos/sampled/positive_text_data.csv', sep='\t')
    posi_statis = posi['d_id'].value_counts()
    posi_statistics = pd.DataFrame({
        'did': posi_statis.index.tolist(),
        'count': posi_statis.values.tolist()
    })
    print(len(posi))
    posi_statistics.to_csv('../../dataset/DP_chaos/sampled/positive_text_data_statistics.txt', sep='\t', index=False)

    nega = pd.read_csv('../../dataset/DP_chaos/sampled/negative_text_data.csv', sep='\t')
    nega_statis = nega['d_id'].value_counts()
    nega_statistics = pd.DataFrame({
        'did': nega_statis.index.tolist(),
        'count': nega_statis.values.tolist()
    })
    nega_statistics.to_csv('../../dataset/DP_chaos/sampled/negative_text_data_statistics.txt', sep='\t', index=False)
    print(len(nega))

def idilize_and_inter_link_file():
    # transform original id to new sequenced id
    # inter file get
    # link file get
    target_path = '../../dataset/OMC-100k/'
    embeddings_path = '../../dataset/OMC-100k-origin/bert_embeddings/'
    users = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k.user', sep='\t')
    nega_users = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k_nega.user', sep='\t')
    items = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k.item', sep='\t')
    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/positive_text_data.csv', sep='\t')
    nega_text_data = pd.read_csv('../../dataset/DP_chaos/sampled/negative_text_data.csv', sep='\t')
    users = users.sort_values(by='department:single_str')
    nega_users = nega_users.sort_values(by='department:single_str')
    items = items.sort_values(by='department:single_str')

    heads = ['profile', 'q', 'dialogue', 'nega_q', 'nega_dialogue']
    embeddings = dict()

    # user_id dict
    user_id_dict = dict(zip(users['user_id:token'].values, range(1, 1 + len(users), 1)))
    nega_user_id_dict = dict(zip(nega_users['user_id:token'].values, range(len(user_id_dict) + 1,
                                                                           len(user_id_dict) + len(nega_users) + 1, 1)))
    # item_id_dict
    item_id_dict = dict(zip(items['item_id:token'].values, range(1, 1 + len(items), 1)))

    # users transform
    for i in range(len(users)):
        users.loc[i, 'user_id:token'] = user_id_dict[users.loc[i, 'user_id:token']]
    users = users.sort_values(by='user_id:token')
    users.to_csv(target_path + 'OMC-100k.user', sep='\t', index=False)
    for i in range(len(nega_users)):
        nega_users.loc[i, 'user_id:token'] = nega_user_id_dict[nega_users.loc[i, 'user_id:token']]
    nega_users = nega_users.sort_values(by='user_id:token')
    nega_users.to_csv(target_path + 'OMC-100k_nega.user', sep='\t', index=False)

    # items transform
    item_columns = list(items.columns)
    for i in range(len(items)):
        items.loc[i, 'item_id:token'] = item_id_dict[items.loc[i, 'item_id:token']]
    items = items.sort_values(by='item_id:token')
    items = items[item_columns]
    items.to_csv(target_path + 'OMC-100k.item', sep='\t', index=False)

    # inter file
    inter = pd.DataFrame({
        'user_id:token': [user_id_dict[int(elem)] for elem in text_data['p_id'].values],
        'item_id:token': [item_id_dict[int(elem)] for elem in text_data['d_id'].values],
        'rating:float': [1] * len(text_data)
    })
    inter.to_csv(target_path + 'OMC-100k.inter', sep='\t', index=False)

    nega_inter = pd.DataFrame({
        'user_id:token': [nega_user_id_dict[int(elem)] for elem in nega_text_data['p_id'].values],
        'item_id:token': [item_id_dict[int(elem)] for elem in nega_text_data['d_id'].values],
        'rating:float': [0] * len(nega_text_data)
    })
    nega_inter.to_csv(target_path + 'OMC-100k_nega.inter', sep='\t', index=False)

    # embeddings
    for head in heads:
        sub_emb = dict()
        t_path = embeddings_path + head + '_embeddings.json'
        with open(t_path, 'r', encoding='utf-8') as f:
            embeddings[head] = json.load(f)
        if head == 'profile':
            for elem in list(embeddings[head].keys()):
                sub_emb[item_id_dict[int(elem)]] = embeddings[head][elem]
        elif head in ['q', 'dialogue']:
            for elem in list(embeddings[head].keys()):
                sub_emb[user_id_dict[int(elem)]] = embeddings[head][elem]
        else:
            for elem in list(embeddings[head].keys()):
                sub_emb[nega_user_id_dict[int(elem)]] = embeddings[head][elem]
        o_path = target_path + 'bert_embeddings/' + head + '_embeddings.json'
        with open(o_path, 'w') as f:
            json.dump(sub_emb, f, ensure_ascii=False)

    # id_dict save
    users_all = []
    users_real_all = []
    for key in user_id_dict.keys():
        users_all.append(user_id_dict[key])
        users_real_all.append(key)
    for key in nega_user_id_dict.keys():
        users_all.append(nega_user_id_dict[key])
        users_real_all.append(key)

    user_id_dict_df = pd.DataFrame({
        'user_id:token': users_all,
        'real_id:token': users_real_all
    })
    user_id_dict_df = user_id_dict_df.sort_values(by='user_id:token')
    user_id_dict_df.to_csv(target_path + 'user_id2real_id.txt', sep='\t', index=False)

    items_all = []
    items_real_all = []
    for key in item_id_dict.keys():
        items_all.append(item_id_dict[key])
        items_real_all.append(key)

    item_id_dict_df = pd.DataFrame({
        'item_id:token': items_all,
        'real_id:token': items_real_all
    })
    item_id_dict_df = item_id_dict_df.sort_values(by='item_id:token')
    item_id_dict_df.to_csv(target_path + 'item_id2real_id.txt', sep='\t', index=False)

    # LDA related user_text file
    user_text = text_data[['p_id', 'query']]
    for i in range(len(user_text)):
        user_text.loc[i, 'p_id'] = user_id_dict[int(user_text.loc[i, 'p_id'])]
    user_text['user_id'] = user_text['p_id']
    user_text = user_text[['user_id', 'query']]
    user_text.to_csv(target_path + 'LDA_related/user_text.csv', sep='\t', index=False)

def doctor_name_extract():
    data_path = '../../dataset/DP_chaos/sampled/text_data.csv'
    data = pd.read_csv(data_path, sep='\t')
    data = data[['d_id', 'profile']]
    data = data.drop_duplicates(keep='first')
    data.reset_index(inplace=True)
    data['name'] = [data.loc[i, 'profile'].split('，')[0] for i in range(len(data))]
    data = data[['d_id', 'name', 'profile']]
    data.to_csv('../../dataset/OMC-100k/doctor2name.txt', sep='\t', index=False)

def docotr_id_name_hospital():
    id_name = pd.read_csv('../../dataset/OMC-100k/doctor_addition_info/doctor2name.txt', sep='\t')
    kg_data = pd.read_csv('../../dataset/OMC-100k/OMC-100k.kg', sep='\t')
    kg_data = kg_data[kg_data['relation_id:token'] == 'doctor.work_in.hospital']
    kg_data = kg_data.reset_index(drop=True)

    id2hos = collections.defaultdict(list)
    for i in range(len(kg_data)):
        id2hos[kg_data.loc[i, 'head_id:token']].append(kg_data.loc[i, 'tail_id:token'])

    hospitals = [id2hos[str(id_name.loc[i, 'd_id'])] if str(id_name.loc[i, 'd_id']) in id2hos.keys() else -1
                 for i in range(len(id_name))]

    id_name['hospital'] = hospitals
    final_results = id_name[['d_id', 'name', 'hospital']]
    final_results.to_csv('../../dataset/OMC-100k/doctor_addition_info/doctor_id_name_hospital.txt', sep='\t', index=False)

def black_list_inter():
    data = pd.read_csv('../../dataset/DP_chaos/sampled/negative_text_data.csv', sep='\t')
    itemid2rid = pd.read_csv('../../dataset/OMC-100k/item_id2real_id.txt', sep='\t')
    id2iid = dict(zip(itemid2rid['real_id:token'].values, itemid2rid['item_id:token'].values))
    userid2rid = pd.read_csv('../../dataset/OMC-100k/user_id2real_id.txt', sep='\t')
    id2uid = dict(zip(userid2rid['real_id:token'].values, userid2rid['user_id:token'].values))
    data = data[['p_id', 'd_id']]
    user_id = []
    item_id = []
    for i in range(len(data)):
        user_id.append(id2uid[data.loc[i, 'p_id']])
        item_id.append(id2iid[data.loc[i, 'd_id']])

    data['user_id:token'] = user_id
    data['item_id:token'] = item_id
    data['rating:float'] = [0] * len(data)
    data = data[['user_id:token', 'item_id:token', 'rating:float']]
    data.to_csv('../../dataset/OMC-100k/OMC-100k_nega.inter', sep='\t', index=False)

def get_att_dis(target, behaviored):
    attention_distribution = []

    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)

    return attention_distribution / torch.sum(attention_distribution, 0)

def get_similar_doctor():
    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/positive_text_data.csv', sep='\t')
    emb_path = '../../dataset/OMC-100k/bert_embeddings/q_embeddings.json'
    id_real_id = pd.read_csv('../../dataset/OMC-100k/user_id2real_id.txt', sep='\t')
    item_real_id = pd.read_csv('../../dataset/OMC-100k/item_id2real_id.txt', sep='\t')
    real_id2id = dict(zip(id_real_id['real_id:token'].values, id_real_id['user_id:token'].values))
    real_id2id_item = dict(zip(item_real_id['real_id:token'].values, item_real_id['item_id:token'].values))
    with open(emb_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    doctors_closed = []
    user_id_box = []
    max_num = 5
    for i in tqdm(range(len(text_data))):
        id = real_id2id[text_data.loc[i, 'p_id']]
        users_same_department = text_data[text_data['department'] == text_data.loc[i, 'department']]
        users_id = users_same_department['p_id'].values
        embs = torch.FloatTensor([embeddings[str(real_id2id[elem])] for elem in users_id])
        similarity = get_att_dis(torch.FloatTensor(embeddings[str(id)]), embs)
        box = [(real_id2id_item[text_data[text_data['p_id'] == users_id[i]]['d_id'].values[0]], similarity[i].tolist())
               for i in range(len(users_id))]
        box = sorted(box, key=operator.itemgetter(1), reverse=True)
        sub_doctors = []
        for did, score in box:
            if len(sub_doctors) >= max_num:
                break
            if did not in sub_doctors:
                sub_doctors.append(did)
        doctors_closed.append(sub_doctors)
        user_id_box.append(id)

    results = pd.DataFrame({
        'user_id': user_id_box,
        'doctor_closed': doctors_closed
    })
    results.to_csv('../../dataset/OMC-100k/user_id_closed_doctors.txt', sep='\t', index=False)

def add_interation():
    data = pd.read_csv('../../dataset/OMC-100k/OMC-100k.inter', sep='\t')
    add_inter = pd.read_csv('../../dataset/OMC-100k/user_id_closed_doctors.txt', sep='\t')
    for i in tqdm(range(len(add_inter))):
        uid = add_inter.loc[i, 'user_id']
        doctors = eval(add_inter.loc[i, 'doctor_closed'])
        for doc in doctors:
            data.loc[len(data), :] = [str(uid), str(doc), str(1)]
    for col in data.columns:
        data[col] = data[col].apply(lambda x: format(int(x)))
    data.to_csv('../../dataset/OMC-100k/OMC-100k_add.inter', sep='\t', index=False)

def filter_patients_based_on_tc():
    text_data = pd.read_csv('../../dataset/DP_chaos/text_data.csv', sep='\t')

    time_wrds = ['急需', '急', '情况严重', '早点', '快', '很快', '严重', '加重', '需要多久', '多久', '什么时候',
                 '多长时间', '现在', '当天']
    time2value = dict(zip(time_wrds, list(range(len(time_wrds), 0, -1))))

    cost_wrds = ['多少钱', '费用', '昂贵', '贵', '便宜', '省钱']
    cost2value = dict(zip(cost_wrds, list(range(len(cost_wrds), 0, -1))))

    pid = []
    department = []
    type_token = []
    index_save = []
    sensitive_words = []
    sensitive_point = []
    for i in tqdm(range(len(text_data))):
        query = text_data.loc[i, 'query']
        # if text_data.loc[i, 'department'] != '心血管内科' or text_data.loc[i, 'd_id'] == 164967473:
        #     continue

        # for wrd in time_wrds:
        #     time_count = 0
        #     time_wrd = []
        #     if query.find(wrd) >= 0:
        #         time_wrd.append(wrd)
        #         time_count += 1
        #         pid.append(text_data.loc[i, 'p_id'])
        #         type_token.append('time')
        #         department.append(text_data.loc[i, 'department'])
        # for wrd in cost_wrds:
        #     if query.find(wrd) >= 0:
        #         pid.append(text_data.loc[i, 'p_id'])
        #         type_token.append('cost')
        #         department.append(text_data.loc[i, 'department'])

        time_count = 0
        time_wrd = []
        for wrd in time_wrds:
            if query.find(wrd) >= 0:
                time_wrd.append(wrd)
                time_count += 1

        cost_count = 0
        cost_wrd = []
        for wrd in cost_wrds:
            if query.find(wrd) >= 0:
                cost_wrd.append(wrd)
                cost_count += 1
        sensitive_words.append([time_wrd, cost_wrd])
        sensitive_point.append([sum([time2value[elem] for elem in time_wrd]),
                                sum([cost2value[elem] for elem in cost_wrd])])
    text_data['sensitive_words'] = sensitive_words
    text_data['sensitive_point'] = sensitive_point

    # result = pd.DataFrame({
    #     'pid': pid,
    #     'department': department,
    #     'token_type': type_token
    # })
    # print(result['department'].value_counts())
    # print(result['token_type'].value_counts())
    #
    # result.to_csv('../../dataset/DP_chaos/pid_qualified.txt', sep='\t', index=False)

    text_data.to_csv('../../dataset/DP_chaos/text_data_full_info.csv', sep='\t', index=False)

def sample_based_on_sensitivity():
    item_feat = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k.item', sep='\t')
    pid_qualified = pd.read_csv('../../dataset/DP_chaos/pid_qualified.txt', sep='\t')
    pid2point = dict(zip(pid_qualified['pid'], [eval(elem)[1] for elem in pid_qualified['sensitive_point']]))

    # doctor fame
    columns = list(item_feat.columns)
    print(columns)
    unused = ['item_id:token', 'department:single_str', 'doctor_title:single_str', 'education_title:single_str',
              'consultation_amount:float', 'active_years:float', 'patient_recommendation_score:float']
    [columns.remove(elem) for elem in unused]

    item_DOOF = dict(zip(item_feat['item_id:token'].values, item_feat[columns].values))

    DOOF = []
    items = list(set(item_feat['item_id:token']))
    for item in items:
        DOOF.append(item_DOOF[item])

    # standard
    scaler = MinMaxScaler()
    values = scaler.fit_transform(DOOF).tolist()
    final_item_DOOF = {items[i]: round(sum(values[i]) / len(values[i]), 4) for i in range(len(items))}

    text_data = pd.read_csv('../../dataset/DP_chaos/text_data_full_info.csv', sep='\t')
    time_value_distribution = [eval(text_data.loc[i, 'sensitive_point'])[0] for i in range(len(text_data))]
    cost_value_distribution = [eval(text_data.loc[i, 'sensitive_point'])[1] for i in range(len(text_data))]
    print(sorted(Counter(time_value_distribution).items(), key=lambda x: x[1], reverse=True))
    print(sorted(Counter(cost_value_distribution).items(), key=lambda x: x[1], reverse=True))
    count = 0
    index_save = []
    time_save = []
    cost_save = []
    blank_save = []
    for i in range(len(text_data)):
        if time_value_distribution[i] > 0 and cost_value_distribution[i] > 0:
            index_save.append(i)
        elif cost_value_distribution[i] > 0:
            cost_save.append(i)
        elif time_value_distribution[i] > 0:
            item = text_data.loc[i, 'd_id']
            if item not in final_item_DOOF.keys():
                continue
            if time_value_distribution[i] >= 14 and cost_value_distribution[i] * (-0.5 / 8) + 0.5 <= final_item_DOOF[item]:
                time_save.append(i)
        else:
            blank_save.append(i)
    print(len(index_save), len(cost_save), len(time_save), len(blank_save))
    core_index = index_save + cost_save + random.sample(time_save, 2500) + random.sample(blank_save, 2000)

    final_data = text_data.loc[core_index, :]
    result = pd.DataFrame({
        'pid': final_data['p_id'],
        'department': final_data['department'],
        'sensitive_point': final_data['sensitive_point']
    })

    # result.to_csv('../../dataset/DP_chaos/pid_qualified.txt', sep='\t', index=False)

def fliter_cost_related_interation():
    text_data = pd.read_csv('../../dataset/DP_chaos/sampled/positive_text_data.csv', sep='\t')
    item_feat = pd.read_csv('../../dataset/OMC-100k-origin/OMC-100k.item', sep='\t')
    pid_qualified = pd.read_csv('../../dataset/DP_chaos/pid_qualified.txt', sep='\t')
    pid2point = dict(zip(pid_qualified['pid'], [eval(elem)[1] for elem in pid_qualified['sensitive_point']]))

    # doctor fame
    columns = list(item_feat.columns)
    print(columns)
    unused = ['item_id:token', 'department:single_str', 'doctor_title:single_str', 'education_title:single_str',
              'consultation_amount:float', 'active_years:float', 'patient_recommendation_score:float']
    [columns.remove(elem) for elem in unused]

    item_DOOF = dict(zip(item_feat['item_id:token'].values, item_feat[columns].values))

    DOOF = []
    items = list(set(item_feat['item_id:token']))
    for item in items:
        DOOF.append(item_DOOF[item])

    # standard
    scaler = MinMaxScaler()
    values = scaler.fit_transform(DOOF).tolist()
    final_item_DOOF = {items[i]: round(sum(values[i]) / len(values[i]), 4) for i in range(len(items))}

    cost_set = []
    DOOF_set = []
    for i in range(len(text_data)):
        pid = text_data.loc[i, 'p_id']
        cost_set.append(pid2point[pid])
        DOOF_set.append(final_item_DOOF[text_data.loc[i, 'd_id']])

    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(cost_set, DOOF_set)
    plt.xlabel("Cost sensitivity of patient")
    plt.ylabel("Fame value of doctor")
    plt.show()

if __name__ == '__main__':
    # data_extract_kg()
    # text_data_process()
    # sample_data()
    # user_process()
    # item_process()
    # idilize_and_inter_link_file()

    # hospital_address_triple()
    # hospital_address_triple_addition()

    # doctor_name_extract()
    # docotr_id_name_hospital()
    # black_list_inter()
    # get_similar_doctor()
    # add_interation()
    # filter_patients_based_on_tc()

    sample_based_on_sensitivity()
    # fliter_cost_related_interation()
