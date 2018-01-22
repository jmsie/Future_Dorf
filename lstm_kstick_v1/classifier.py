import os
import sys
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
from sklearn import mixture
from fetch import Future

def bezier(data, bezier_times=10, show_figure=False):
    '''use bezier curve to redraw close data'''
    bezier = copy.copy(data[:,3])
    for j in range(bezier_times):
        for i in range(bezier.shape[0]-4):
            for tt in range(1,4):
                t = tt*0.25
                bezier[i+tt] = bezier[i]*(1-t)**4 + 4*bezier[i+1]*t*(1-t)**3 + 6*bezier[i+2]*(t**2)*(1-t)**2 + 4*bezier[i+3]*(t**3)*(1-t) + bezier[i+4]*(t**4)
    if show_figure==True:
        plt.plot(range(bezier.shape[0]), bezier, 'r--')
        plt.plot(range(bezier.shape[0]), data[:,3])
        plt.show()
    return bezier

def calculate_slope(bezier, poly_degree=5, times=20, show_figure=False):
    '''compact consolidation time'''
    warnings.simplefilter('ignore', np.RankWarning)
    #warnings.simplefilter('ignore', np.RuntimeWarning)
    #np.warnings.filterwarnings('ignore')
    slp = []
    for i in range(1, len(bezier)):
        s = max(i-times, 0)
        z = np.polyfit(range(s,i), bezier[s:i], poly_degree).flatten()
        sl = np.zeros(poly_degree)
        for j in range(poly_degree):
            sl[j] = z[j]*(poly_degree-j)
        p = np.poly1d(sl)
        slp.append(p(i-1))
        if i == len(bezier)-1:
            slp.append(p(i))

    if show_figure==True:
        plt.plot(range(len(bezier)), bezier, 'r--')
        plt.plot(range(len(slp)), slp)
        plt.show()
    return slp

def merge_kstick(data):
    return [data[0, 0], max(data[:, 1]), min(data[:, 2]), data[-1, 3]]

class GMMClassifier:
    def __init__(self, data, consolidation_slope=0.5, Klength_cluster=7, Ktype_cluster=6, Ktime_cluster=5):
        self.data = data
        self.consolidation_slope = consolidation_slope
        self.Klength_cluster = Klength_cluster
        self.Ktype_cluster = Ktype_cluster
        self.Ktime_cluster = Ktime_cluster
        self.Kstick_output_number = 0
        self.Kstick_input_number = 0

    def _train_kstick_time(self):
        slp_lst = calculate_slope(bezier(self.data))
        low_slp = []
        slp_count = 0
        for slp in slp_lst:
            if abs(slp) <= self.consolidation_slope:
                slp_count += 1
            else:
                if slp_count > 0:
                    low_slp.append(slp_count)
                slp_count = 0

        low_slp = np.array(low_slp).reshape((len(low_slp), 1))
        self.gmm_Ktime = mixture.GaussianMixture(n_components=self.Ktime_cluster, max_iter=2000)
        self.gmm_Ktime.fit(low_slp)

        cluster_id = self.Kstick_output_number
        for i in range(self.gmm_Ktime.means_.shape[0]):
            for j in range(self.gmm_Ktype.means_.shape[0]):
                for k in range(3):
                    self.cluster[(j, k, i+1)] = cluster_id
                    cluster_id += 1

        self.Kstick_input_number = len(self.cluster)

    def predict_kstick_time(self, time):
        return self.gmm_Ktime.predict(time)[0]+1

    def predict_kstick_type(self, data):
        x = np.zeros((1, 3))
        sort = sorted(data)
        kstick_cluster = self.gmm_Klength.predict(sort[-1]-sort[0])
        kstick_scale = self.gmm_Klength.means_[kstick_cluster]/float(sort[-1]-sort[0]+10e-6)
        x[0, 0] = (sort[1]-sort[0])*kstick_scale
        x[0, 1] = (sort[2]-sort[1])*kstick_scale
        x[0, 2] = (sort[3]-sort[2])*kstick_scale
        return self.gmm_Ktype.predict(x)[0]

    def predict_kstick_color(self, data):
        if data[-1]==data[0]:
            return 0
        elif data[-1]>data[0]:
            return 1
        else:
            return 2

    def _train_kstick_length(self):
        train_x = np.zeros((self.data.shape[0], 1))
        for idx in range(self.data.shape[0]):
            sort = sorted(self.data[idx])
            train_x[idx, 0] = sort[-1]-sort[0]
        
        self.gmm_Klength = mixture.GaussianMixture(n_components=self.Klength_cluster, max_iter=5000)
        self.gmm_Klength.fit(train_x)
        #print(self.gmm_Klength.means_)

    def _train_kstick_type(self):
        train_x = np.zeros((self.data.shape[0], 3))
        for idx in range(self.data.shape[0]):
            sort = sorted(self.data[idx])
            kstick_cluster = self.gmm_Klength.predict(sort[-1]-sort[0])
            kstick_scale = self.gmm_Klength.means_[kstick_cluster]/float(sort[-1]-sort[0]+10e-6)
            train_x[idx, 0] = (sort[1]-sort[0])*kstick_scale
            train_x[idx, 1] = (sort[2]-sort[1])*kstick_scale
            train_x[idx, 2] = (sort[3]-sort[2])*kstick_scale

        self.gmm_Ktype = mixture.GaussianMixture(n_components=self.Ktype_cluster, max_iter=20000)
        self.gmm_Ktype.fit(train_x)

        self.cluster = {}
        cluster_id = 0
        ## type cluster, (equal, rise, fall), minute
        for idx in range(self.gmm_Ktype.means_.shape[0]):
            self.cluster[(idx, 0, 0)] = cluster_id
            cluster_id += 1
            self.cluster[(idx, 1, 0)] = cluster_id
            cluster_id += 1
            self.cluster[(idx, 2, 0)] = cluster_id
            cluster_id += 1

        self.Kstick_output_number = len(self.cluster)
        #print(self.Kstick_output_number)

    def fit(self):
        self._train_kstick_length()
        self._train_kstick_type()
        self._train_kstick_time()

    def transform(self, data, number_data=20):
        #min_id = max(current_id-number_data*2, 0)
        slp_lst = calculate_slope(bezier(data, bezier_times=5, show_figure=False), times=number_data)
        #print(len(slp_lst), len(data))
        low_slp_count = 0
        kstick_idx = len(slp_lst)-1
        kstick_lst = []
        test = []
        while len(kstick_lst)<number_data and kstick_idx>=0:
            if abs(slp_lst[kstick_idx]) <= self.consolidation_slope:
                low_slp_count += 1
            else:
                #print(low_slp_count)
                if low_slp_count > 0:
                    kstick = merge_kstick(data[kstick_idx+1:kstick_idx+low_slp_count+1])
                    kstick_type = self.predict_kstick_type(kstick)
                    kstick_time = self.predict_kstick_time(low_slp_count)
                    kstick_lst.append(self.cluster[(kstick_type, self.predict_kstick_color(kstick), kstick_time)])
                    if len(kstick_lst)==number_data:
                        continue
                    #test.append((kstick_type, self.predict_kstick_color(kstick), kstick_time))
                
                kstick = data[kstick_idx]
                #print(kstick)
                kstick_type = self.predict_kstick_type(kstick)
                kstick_time = 0
                kstick_lst.append(self.cluster[(kstick_type, self.predict_kstick_color(kstick), kstick_time)])
                #test.append((kstick_type, self.predict_kstick_color(kstick), kstick_time))
                low_slp_count = 0

            kstick_idx -= 1

        #print(kstick_lst[::-1])
        if len(kstick_lst)<number_data:
            return False
        else:
            return kstick_lst[::-1]

    def predict(self, data):
        stick_type = self.predict_kstick_type(data)
        stick_color = self.predict_kstick_color(data)
        stick_time = 0
        return self.cluster[(stick_type, stick_color, stick_time)]

# if __name__ == "__main__":
#     #future = RNN()
#     #future.train_optim()
#     test = Future('http://dorfcapital.com/asset/data/MXF1-Minute-Trade.txt')
#     data = test.get_Kstick_data(time_range=5, data_length=15000)
#     Kstick = GMMClassifier(data)
#     Kstick.fit()
#     Kstick.transform(data[30:50])
