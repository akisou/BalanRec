import re
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import poisson
from tqdm import tqdm

def datetime_transform(data):
    complete_pattern = r"(\d{4}.\d{1,2}.\d{1,2})"
    part_pattern = r"(\d{1,2}.\d{1,2})"
    null_pattern = '-1'
    special_time = {'今天': '2022.11.18', '昨天': '2022.11.17'}

    query_time = []
    for i in range(len(data)):
        fr_time = data.loc[i, 'first_response_time']
        if fr_time == null_pattern:
            continue

        if re.search(complete_pattern, fr_time):
            fr_time = fr_time
        elif re.search(part_pattern, fr_time):
            fr_time = '2022.' + fr_time
        else:
            fr_time = special_time[fr_time]

        data.loc[i, 'first_response_time'] = int(time.mktime(datetime.strptime(fr_time, '%Y.%m.%d').timetuple()))

        data.loc[i, 'query_time'] = int(time.mktime(datetime.strptime(data.loc[i, 'query_time'], '%Y-%m-%d').timetuple()))

        print(data.loc[i, :])

    # data.to_csv('../dataset/first_response_times_transformed.csv', sep='\t')
    return data

def fit_function(k, lamb):
    # The parameter lamb will be used as the fit parameter
    return poisson.pmf(k, lamb)

def draw_distribution(data):
    start_time = 1388505600
    data = data[(data['first_response_time'] != -1) & (data['first_response_time'] != '-1')]
    data = data.reset_index(drop=True)
    data.to_csv('../../dataset/first_response_times_transformed_mk.csv', sep='\t', index=0)
    r_q = []
    q_q0 = []
    for i in tqdm(range(len(data))):
        # print(data.loc[i, :])
        r_q.append(int(data.loc[i, 'first_response_time']) - int(data.loc[i, 'query_time']))
        q_q0.append(int(data.loc[i, 'query_time']) - start_time)

    plt.figure(dpi=120)
    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    g = sns.distplot(r_q,
                     hist=True,
                     kde=False,
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',
                              },
                     fit=poisson,  #
                     color='#098154', )
    _, p = poisson.fit(r_q).chisq_test()
    print('q_r', p)

    plt.figure(dpi=120)
    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    g = sns.distplot(q_q0,
                     hist=True,
                     kde=False,
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',
                              },
                     fit=poisson,  #
                     color='#098154', )

    _, p = poisson.fit(q_q0).chisq_test()
    print('q_q0', p)
    # bins = np.arange(20) - 0.5
    # entries, bin_edges, patches = plt.hist(q_q0, bins=bins, density=True, label='Data')
    #
    # # calculate bin centers
    # middles_bins = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    #
    # # fit with curve_fit
    # parameters, cov_matrix = curve_fit(fit_function, middles_bins, entries)
    #
    # # plot poisson-deviation with fitted parameter
    # x_plot = np.arange(0, 15)
    #
    # plt.plot(
    #     x_plot,
    #     fit_function(x_plot, *parameters),
    #     marker='D', linestyle='-',
    #     color='red',
    #     label='Fit result',
    # )
    # plt.legend()
    # plt.show()



if __name__ == "__main__":
    data = pd.read_csv('../../dataset/first_response_times_transformed.csv', sep='\t')
    # data_transformed = datetime_transform(data)
    draw_distribution(data)
