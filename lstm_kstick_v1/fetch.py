import requests
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import datetime as datetime
from matplotlib.finance import candlestick2_ohlc
from sklearn.neighbors import KernelDensity


class Future:
    def __init__(self, url, data_type='minute'):
        self.__url__ = url
        self.data_type = data_type
        self.data = []
        self.data_frame = []
        self.table = ""

    def fetch(self, dump_txt=False):
        output = open('test.txt', 'w')
        r = requests.get(self.__url__, stream=True)
        count = 0
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if dump_txt==True:
                    output.write(decoded_line)
                    output.write('\n')
                if count!=0:
                    self.data.append(re.split(',', decoded_line))
                else:
                    self.data_frame = re.split(',', decoded_line)
                count += 1
        self.data = np.array(self.data)
        output.close()
        #print(self.data[0:,1])

    def fetch_txt(self):
        self.table = pd.read_table('test.txt', sep=',')

    def get_Kstick_data(self, time_range=1, data_length=100):
        self.fetch_txt()
        data = np.zeros((data_length, 4))
        for i in range(data_length):
            data[i][2] = min(self.table['Low'][i*time_range:(i+1)*time_range])
            data[i][1] = max(self.table['High'][i*time_range:(i+1)*time_range])
            data[i][0] = self.table['Open'][i*time_range]
            data[i][3] = self.table['Close'][(i+1)*time_range-1]
        return data

    def plot_Kstick(self, array):
        fig, ax = plt.subplots()
        candlestick2_ohlc(ax, array[:,0], array[:,1], array[:,2], array[:,3], colorup = 'r', colordown='g', width=1, alpha=1)
        plt.show()

    # def plot_relative(self):
    #     small = self.table[0:200][['Open', 'High', 'Low', 'Close']]
    #     _ = pd.scatter_matrix(small)
    #     plt.show()

    def get_MA(self, s, e, time):
        self.fetch_txt()
        array = np.array(self.table[s:e].reset_index()[['Open', 'High', 'Low', 'Close', 'TotalVolume']])
        arr = np.zeros((int(array.shape[0]/time), array.shape[1]))
        for n in range(arr.shape[0]):
            arr[n] = np.mean(array[n*time:(n+1)*time], axis=0)
            arr[n,-1] = np.sum(array[n*time:(n+1)*time,-1], axis=0)

        return arr

    def get_TR(self, s, e, time=1, avg_time=20, show_figure=False):
        self.fetch_txt()
        array = np.array(self.table[s:e].reset_index()[['Open', 'High', 'Low', 'Close', 'TotalVolume']])
        tmp_length = int(array.shape[0]/time)
        tmp = np.zeros((tmp_length, 4))
        for i in range(tmp_length):
            tmp[i,0] = array[i*time,0]
            tmp[i,1] = max(array[i*time:(i+1)*time,1])
            tmp[i,2] = min(array[i*time:(i+1)*time,2])
            tmp[i,3] = array[(i+1)*time-1,3]
        tr = np.zeros((tmp_length, 1))
        for i in range(1, tmp_length):
            tr[i] = max(tmp[i,1]-tmp[i,2], tmp[i,1]-tmp[i-1,3], tmp[i-1,3]-tmp[i,2])
        tr[0] = tr[1]

        if show_figure==True:
            plt.figure(1)
            plt.plot(range(len(tr)), tr)
            plt.figure(2)
            plt.plot(range(len(tr)), tmp, 'r--')
            plt.show()

    def bezier(self, s, e, time=1):
        self.fetch_txt()
        array = np.array(self.table[s:e].reset_index()[['Open', 'High', 'Low', 'Close', 'TotalVolume']])
        bezier = np.zeros((array.shape[0], 1))
        for i in range(bezier.shape[0]):
            bezier[i] = array[i,3]

        for j in range(10):
            for i in range(bezier.shape[0]-4):
                for tt in range(1,4):
                    t = tt*0.25
                    bezier[i+tt] = bezier[i]*(1-t)**4 + 4*bezier[i+1]*t*(1-t)**3 + 6*bezier[i+2]*(t**2)*(1-t)**2 + 4*bezier[i+3]*(t**3)*(1-t) + bezier[i+4]*(t**4)

        slp = []
        for i in range(1,len(bezier)):
            s = max(i-20, 0)
            z = np.polyfit(range(s,i), bezier[s:i], 5).flatten()
            sl = np.zeros(5)
            for j in range(5):
                sl[j] = z[j]*(5-j)
            p = np.poly1d(sl)
            slp.append(p(i-1)*10+5450)

        #plt.figure(1)
        plt.plot(range(len(slp)), slp, 'r--')
        line1 = [5450]*len(bezier)
        line2 = [5445]*len(bezier)
        line3 = [5455]*len(bezier)

        #plt.figure(2)
        plt.plot(range(len(bezier)), bezier)
        plt.plot(range(len(bezier)), line1, 'b')
        plt.plot(range(len(bezier)), line2, 'b')
        plt.plot(range(len(bezier)), line3, 'b')
        #plt.figure(2)
        #plt.plot(range(len(bezier)), array[:,3], 'r--')
        plt.show()


#test = Future('http://dorfcapital.com/asset/data/MXF1-Minute-Trade.txt')
#test.get_TR(0,1000, show_figure=True)
#test.bezier(0,1000)
#test.fetch(True)
#test.fetch_txt()
#data = test.get_Kstick_data(time_range=7, data_length=20000)
#test.plot_Kstick(data)
#test.plot_relative()
