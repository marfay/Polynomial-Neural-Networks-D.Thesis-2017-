from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.matplotlib.backend_kivy import FigureManagerKivy, show, new_figure_manager, NavigationToolbar2Kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import matplotlib.pyplot as plt
import threading
#from kivy.clock import Clock
import time
import neural_network as nn
import numpy as np
import os
import load_ene as ee
#from kivy.uix.codeinput import CodeInput


class MainWindow(BoxLayout):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.LIST_OF_GRAPHS = []
        self.LIST_OF_ACTION_BARS = []
        self.constant = 0
        self.load_data()
        self.percentage = '0%'
        self.yn = np.zeros(10)
        self.normalize_data = 0
        self.cancel = False # constanta pro preruseni vypoctu -> viz gui/button cancel
        self.sampling = 96 # vzorkovani data -> den 96 vzorku


    def thread_code_execution(self): # Spusteni vypoctu z kodu ve Free modu ve vlastnim vlakne
        threading.Thread(target=self.code_execution).start()

    def thread_code_execution_yr(self): # Spusteni kodu ve Free modu ve vlastnim vlakne - zobrazeni yr
        threading.Thread(target=self.code_execution_yr).start()



    def thread_graph(self):
        threading.Thread(target = self.graph_outside).start()

    def graph_outside(self): # Nema tu co delat - ale spustenim day_examination.py ziskam graf z uvahy, takze to tu necham
        os.system("day_examination.py")

    """
    -----------------DATA LOGIC
    """
    def load_data(self): # nahrani dat ze souboru (spusteno pri zapnuti programu -> viz init)
        self.ddays = (np.loadtxt('log_days.txt'))
        self.ttime = (np.loadtxt('time.txt'))
        #self.ttime = np.load('the_time.npy')
        self.X = np.loadtxt('ENE_data.txt') # data obsahujici i umela data
        self.date_X, self.time_X = ee.load_data()
        #self.weather= np.loadtxt()


    def which_day(self): # viz polozka data, zjisti kolikaty den koresponduje se zadanym datem
        n = 0
        for i in self.date_X:
            if i == self.ids.what_day.text:
                break
            n += 1
        self.ids.wh_day.text = str(n/self.sampling)+'.'


    """
    Neural calculations
    """
    def code_execution_yr(self): # funkce pro ziskani yr z Free modu
        loc = {}
        try:
            exec (self.ids.code_it.text, {}, loc) # Vytahnuti promenych ze skriptovaciho okenka free mode do lokalniko slovniku loc
            yr = loc['yr']
            self.normalize_data = np.amax(yr)  # ziskani konstanty pro normalizaci dat (nejvetsi cislo z namerenych dat)
            self.ax.cla() # vycisteni grafu
            self.graph, = self.ax.plot([], [], 'k') #inicializace plotu (yr)
            self.graph2, = self.ax.plot([], [], 'g') #inicializace plotu (yn)
            self.graph3, = self.ax.plot([], [], 'r', linewidth=1, linestyle='--') #inicializace plotu (e)
            x_axis = np.arange(yr.shape[0]) # data X (posloupnost cisel korespodujici s yn,yr a e)
            self.graph.set_xdata(x_axis)
            self.graph.set_ydata(yr)
            self.ax.set_ylim(np.amin(yr) - np.abs(np.amin(yr))*0.3, np.abs(np.amax(yr)) * 1.3) # nastaveni os
            self.ax.set_xlim(0, x_axis[-1]) # nastaveni os
            self.canvas.draw() # prekreslit
        except:
            print('error')

    def code_execution(self): # viz funkce code_execution_yr
        loc = {}
        try:
            exec (self.ids.code_it.text, {}, loc)
            X = loc['X']
            yr = loc['yr']
            nodes = loc['nodes']
            learning_rate = loc['learning_rate']
            re_learn = loc['relearn_every']
            epoch1 = loc['epoch1']
            epoch2 = loc['epoch2']
            p = loc['p']
            win = loc['win']
            self.normalize_data = np.amax(yr)
            self.ids.graph_accordion.collapse = True # po zmacknuti RUN ve free modu se prepne z grafu na info
            self.ids.info.collapse = False # po zmacknuti RUN ve free modu se prepne z grafu na info

            self.ids.jacobian_calcul.text = '-' # text v info
            self.ids.jacobian_calcul_time.text = '-'
            self.ids.calcul_time.text = '-'
            self.ids.finished.text = '-'


            yr = yr/self.normalize_data # normalizace dat
            self.code_yr = yr
            self.yr = yr # yr je globalni

            self.code_x_axis=np.arange(yr.shape[0])

            tn = self.ids.neural_type.text # typ neuornu
            mu = learning_rate

            # VYBER METODY A DRUHU NEURONU
            if self.ids.neural_method.text == 'CGD':
                threading.Thread(target=self.CGD_batch,
                                 kwargs={'type': tn, 'X': X, 'yr': yr, 'p': p, 'win': win, 're_learn': re_learn,
                                         'epoch1': epoch1, 'epoch2': epoch2}).start()
            if self.ids.neural_method.text == 'LM':
                if self.ids.neural_type.text == 'MLP':
                    threading.Thread(target=self.batch_MLP,
                                     kwargs={'method': 'LM', 'nodes': nodes, 'X': X, 'yr': yr, 'p': p, 'win': win,
                                             're_learn': re_learn, 'epoch1': epoch1, 'epoch2': epoch2,
                                             'learning_rate': mu}).start()
                else:
                    threading.Thread(target=self.LM_batch,
                                     kwargs={'type': tn, 'X': X, 'yr': yr, 'p': p, 'win': win, 're_learn': re_learn,
                                             'epoch1': epoch1, 'epoch2': epoch2, 'learning_rate': mu}).start()
            if self.ids.neural_method.text == 'NGD' or self.ids.neural_method.text == 'GD':
                if self.ids.neural_type.text == 'MLP':
                    threading.Thread(target=self.XGD_sample_MLP,
                                     kwargs={'type': self.ids.neural_method.text, 'nodes': nodes, 'X': X, 'yr': yr,
                                             'p': p, 'win': win, 're_learn': re_learn,
                                             'epoch1': epoch1, 'epoch2': epoch2, 'learning_rate': mu}).start()
                else:
                    threading.Thread(target=self.XGD_sample_HONU,
                                     kwargs={'method': self.ids.neural_method.text, 'type': tn, 'X': X, 'yr': yr,
                                             'p': p,
                                             'win': win, 're_learn': re_learn,
                                             'epoch1': epoch1, 'epoch2': epoch2, 'learning_rate': mu}).start()
            self.ids.result_code_button.disabled = False

        except:
            print("Error in the inserted CODE")

    def get_params_and_start(self):
        # Vypocet v polozce Prediction
        self.ids.graph_accordion.collapse = True
        self.ids.info.collapse = False

        self.ids.jacobian_calcul.text = '-'
        self.ids.jacobian_calcul_time.text = '-'
        self.ids.calcul_time.text = '-'
        self.ids.finished.text = '-'


        sd = self.sampling # samples per day
        p=int(self.ids.number_prediction.text) # ziskani dat z menu jako predice, neurony.....
        nodes = int(self.ids.number_nodes.text)
        data__type = int(self.ids.data_type.value)
        try:
            sd1 = int(self.ids.sda.text) * sd # z polozky Data ziskam interval dnu od sd1 do sd2
            sd2 = int(self.ids.sdb.text) * sd
        except: # Toto jsem udelal, protoze jsem nevedel, jaky by byli dusledky, kdyby uzivatel zvolil prilis kratky interval dat pro predikci
            sd1 = 3 * sd
            sd2 = (3 + 21) * sd
        if sd2 < (sd1 + 14 * sd): # min pocet dnu 21 pro vypocet
            sd2 = sd1 + 21 * sd
        #Load data
        X = self.X[sd1:sd2,data__type]


        try: # Coarse graning
            if self.ids.cgrain_on_off.active == True and int(self.ids.radius.text) > 0:
                X = self.coarse_grain_data(X)
        except:
            print('cislo musi byt cele a kladne')
        self.normalize_data = np.amax(X)
        X = X/(1.0*self.normalize_data)
        self.X_last = X # globalni promenna
        D = self.ddays[sd1:sd2]
        T = self.ttime[sd1:sd2]
        yr = 1.0*X
        win = int(self.ids.number_win.text) * sd
        re_learn = int(self.ids.number_relearn_every.text)*sd
        epoch1=int(self.ids.number_epoch1.text)
        epoch2=int(self.ids.number_epoch2.text)
        tn = self.ids.neural_type.text
        mu = float(self.ids.number_mu.text)
        #print(mu)
        """
        Input vector FUSION
        """
        # tvorba vstupni matice
        n_en = int(self.ids.dynamic_samples_en.text)
        g_en = int(self.ids.gap_in_dynamic_samples_en.text)
        n_D = int(self.ids.dynamic_samples_days.text)
        g_D = int(self.ids.gap_in_dynamic_samples_days.text)
        n_T = int(self.ids.dynamic_samples_time.text)
        g_T = int(self.ids.gap_in_dynamic_samples_time.text)
        # Trochu zmatecne, nicmnene jsem chtel otestovat kvalitu predikce bez namerenych hodnot ve vstupu -> Celkem hruza
        if n_en ==0: # tvorba vstupni matice bez namerenych hodnot (zada se 0 do textoveho pole)
            X = nn.force_recurrent_gap(T, n_T, g_T, 0)
        else:
            X = nn.force_recurrent_gap(X,n_en,g_en,0)
            T = nn.force_recurrent_gap(T, n_T,g_T, 0)
            X = np.hstack((X, T))
        D = nn.force_recurrent_gap(D, n_D,g_D, 0)
        X = np.hstack((X, D))
        self.yr=yr
        self.ids.show_button.disabled = False
        # vyber METODY A NEURONU pro vypocet
        if self.ids.neural_method.text == 'CGD':
            threading.Thread(target=self.CGD_batch,kwargs={'type': tn,'X': X,'yr':yr,'p':p,'win':win,'re_learn':re_learn,'epoch1':epoch1,'epoch2':epoch2}).start()
        if self.ids.neural_method.text == 'LM':
            if self.ids.neural_type.text =='MLP':
                threading.Thread(target=self.batch_MLP,kwargs={'method': 'LM','nodes': nodes ,'X': X, 'yr': yr, 'p': p, 'win': win, 're_learn': re_learn,'epoch1': epoch1, 'epoch2': epoch2, 'learning_rate': mu}).start()
            else:
                threading.Thread(target=self.LM_batch,kwargs={'type': tn,'X': X,'yr':yr,'p':p,'win':win,'re_learn':re_learn,'epoch1':epoch1,'epoch2':epoch2,'learning_rate':mu}).start()
        if self.ids.neural_method.text == 'NGD' or self.ids.neural_method.text == 'GD':
            if self.ids.neural_type.text =='MLP':
                threading.Thread(target=self.XGD_sample_MLP,kwargs={'type': self.ids.neural_method.text,'nodes':nodes, 'X': X, 'yr': yr, 'p': p, 'win': win, 're_learn': re_learn,
                                     'epoch1': epoch1, 'epoch2': epoch2, 'learning_rate': mu}).start()
            else:
                threading.Thread(target=self.XGD_sample_HONU,
                                 kwargs={'method': self.ids.neural_method.text,'type': tn, 'X': X, 'yr': yr, 'p': p,
                                         'win': win, 're_learn': re_learn,
                                         'epoch1': epoch1, 'epoch2': epoch2, 'learning_rate': mu}).start()




            #def XGD_sample_MLP(self, type, nodes, X, yr, p, win, re_learn, epoch1, epoch2,
            #               learning_rate):  # NGD = 0.005, GD = NGD/10

    ##### PRO KOMENTARE -> ALGORITMY VYPOCTU VIZ  !!! neural_network_functions.py  !!!

    def CGD_batch(self,type, X, yr, p, win, re_learn, epoch1, epoch2):
        # VIZ neural_network_functions.py
        L = int(yr.shape[0])
        X = nn.input_shape(X)
        yr = nn.output_shape(yr)
        """The dimension of neuron and all matrixes like weights and so on
        depends on number of columns of input vector"""
        DIM = nn.input_dim(X[0, :])
        """Creating Neuron object with particular dimension"""
        if type == 'CNU':
            XNU_B = nn.HONU().CNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_B.col_W_from_W(XNU_B.W_CNU_init)
        if type == 'QNU':
            XNU_B = nn.HONU().QNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_B.col_W_from_W(XNU_B.W_QNU_init)
        if type == 'LNU':
            XNU_B = nn.HONU().LNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_B.col_W_from_W(XNU_B.W_LNU_init)
        """Creating Method object(BackPropagation) with particular method"""





        #### Learning Entropy# ---------------------------------1-------------------------------------------
        ole = int(self.ids.OLES.text)
        memory = int(self.ids.memory.text)
        Wm, self.EA, self.EAP, OLEs = nn.Learning_entropy(col_W, ole, L, p, memory)
        alfas = self.ids.alfas.text
        alfas = np.fromstring(alfas, dtype=float, sep=' ')
        multiply = float(self.ids.multiply.text)
        alfas = alfas/multiply
        allow_e = self.ids.entr_allow.active
        ## MAE RMSE
        self.e = np.zeros(L + p)
        self.MAE = np.zeros(L+p)
        self.RMSE = np.zeros(L+p)
        ### BREAK Retraining
        try:
            intervals = np.fromstring(self.ids.break_intervals.text, dtype=int, sep=' ')
            intervals = intervals.reshape((intervals.shape[0]/2, 2))
        except:
            intervals= np.array([1, 2])


        LE_constant = 1

        ##### -----------------------------------1-----------------------------------------




        BP = nn.Backpropagation()
        CGD = BP.CGD()
        yn = np.zeros(L + p)
        self.yn = np.zeros(L + p)
        self.ids.jacobian_calcul.text = 'being calculated'
        self.ids.jacobian_calcul.color = [1, 0, 0, 1]
        start = time.time()
        JAC = XNU_B.Jacobian(X[:, :])
        end = time.time() - start
        self.ids.jacobian_calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.jacobian_calcul.text = 'calculation finished'
        self.ids.jacobian_calcul.color = [0, 1, 0, 1]

        J = JAC[:win, :]
        A = CGD.A(J)
        b = CGD.b(yr[p:p + win], J)
        re = CGD.re(b, A, col_W)
        pp = re
        start = time.time()


        for i in range(epoch1):
            col_W, pp, re = CGD.CGD(A, col_W, re, pp)
        """Sliding window"""
        for k in range(win + p, L):
            """Re-Learn every x samples"""
            #print(re_learn)
            if k % (re_learn) == 0 and k >= win + p + 1:  # Retrain
                # XX = X[k-win-p+1:k-p+1,:]
                # J = XNU_B.Jacobian(XX)
                J = JAC[k - win - p + 1:k - p + 1, :]
                A = CGD.A(J)
                b = CGD.b(yr[k - win + 1:k + 1], J)
                re = CGD.re(b, A, col_W)
                pp = re
                self.ids.finished.text = str('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
                self.ids.finished.color=[1,0,0,1]
                #print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')

                break_constant = 0# -----------------------------------------------2-----------------------------
                if self.ids.break_pretrain_s.active == True:
                    for i in intervals:

                        if k >= i[0] and k<i[1]+win+p:
                            break_constant = 1
                LE_constant = 1 # for right calculation of LE


                for epoch in range(epoch2):
                    if break_constant == 1:  # -------------------------------------2---------------------------------------
                        break
                    col_W, pp, re = CGD.CGD(A, col_W, re, pp)

            """For prediction we only need one row of samples as they include all reccurent values"""
            # XX = X[k:k+1,:] # staci X[k-1,:], takto je temporary.shape = 1,
            J = JAC[k:k + 1, :]
            # J = XNU_B.Jacobian(XX)
            temporary = XNU_B.Y(J, col_W)
            yn[k + p] = temporary[-1]  # PROC k+p-1



            if allow_e == True:# -------------------------------------3---------------------------------------
                if LE_constant == 1:
                    Wm = nn.Wm_operation(Wm, col_W)
                    LE_constant = 0
                self.EA[k, :] = nn.fcnEA(Wm, alfas, OLEs)  # LE # spravne tohle ma byt v if
                self.EAP[k, :] = self.EAP[k - 1, :] + self.EA[k, :]
                #else:
                    #self.EA[k, :] = 0  # LE
                    #self.EAP[k, :] =self.EAP[k - 1, :] +self.EA[k, :]
            # cancel calc.
            if self.cancel == True:# ---------------------------------3-------------------------------------------
                break


        end = time.time() - start
        self.ids.calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.finished.text = str('Finished: ' + str('100' + '%'))
        self.ids.finished.color = [0, 1, 0, 1]



        self.e[:L] = yr[:L]-yn[:L] # ------------------------------------------4----------------------------------
        self.yn = yn
        for k in range(win+p,L): # Smycka na konci at nezpomaluje vypocet
            self.MAE[k] = np.mean(np.abs(yr[win:k] - yn[win:k]))
            self.RMSE[k] = (np.mean((yr[win:k] - yn[win:k]) ** 2)) ** 0.5# ------------------4----------------------------------------------------------


        return yn

    def LM_batch(self,type, X, yr, p, win, re_learn, epoch1, epoch2, learning_rate):
        L = int(yr.shape[0])
        X = nn.input_shape(X)
        yr = nn.output_shape(yr)
        """The dimension of neuron and all matrices like weights and so on
        depends on number of columns of input vector"""
        DIM = nn.input_dim(X[0, :])
        """Creating Neuron object with particular dimension"""
        if type == 'CNU':
            XNU_B = nn.HONU().CNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_B.col_W_from_W(XNU_B.W_CNU_init)
        if type == 'QNU':
            XNU_B = nn.HONU().QNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_B.col_W_from_W(XNU_B.W_QNU_init)
        if type == 'LNU':
            XNU_B = nn.HONU().LNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_B.col_W_from_W(XNU_B.W_LNU_init)
        """Creating Method object(BackPropagation) with particular method"""
        BP = nn.Backpropagation()
        yn = np.zeros(L + p)
        yn_temporary = np.zeros(L + p)
        SSE = 0
        self.ids.jacobian_calcul.text = 'being calculated'
        self.ids.jacobian_calcul.color = [1, 0, 0, 1]
        start = time.time()
        JAC = XNU_B.Jacobian(X[:, :])
        end = time.time()-start
        # J = XNU_B.Jacobian(X[:win,:])
        J = JAC[:win, :]
        self.ids.jacobian_calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.jacobian_calcul.text = 'calculation finished'
        self.ids.jacobian_calcul.color = [0,1,0,1]
        self.EA = 0

        #### LE# ---------------------------------1-------------------------------------------
        ole = int(self.ids.OLES.text)
        memory = int(self.ids.memory.text)
        Wm, self.EA, self.EAP, OLEs = nn.Learning_entropy(col_W, ole, L, p, memory)
        alfas = self.ids.alfas.text
        alfas = np.fromstring(alfas, dtype=float, sep=' ')
        multiply = float(self.ids.multiply.text)
        alfas = alfas / multiply
        allow_e = self.ids.entr_allow.active
        ## MAE RMSE
        self.e = np.zeros(L + p)
        self.MAE = np.zeros(L + p)
        self.RMSE = np.zeros(L + p)
        ### BREAK Retraining
        try:
            intervals = np.fromstring(self.ids.break_intervals.text, dtype=int, sep=' ')
            intervals = intervals.reshape((intervals.shape[0] / 2, 2))
        except:
            intervals = np.array([1, 2])
        LE_constant = 1
            ##### -----------------------------------1-----------------------------------------


        for i in range(epoch1):
            yn[p:p + win] = XNU_B.Y(J, col_W)
            col_W,dw = BP.LM(yr[p:p + win], yn[p:p + win], learning_rate, col_W,
                          J)  # y_target,y_neuron, learning rate, col_W, Jacobian(col_X)
            e = yr[p:p+win]-yn[p:p + win]
            SSE=np.append(SSE,np.dot(e,e))
        yn = np.zeros(L + p)
        """Sliding window"""
        for k in range(win + p, L):
            """Re-Learn every x samples, here we relearn every day"""
            if k % (re_learn) == 0 and k >= win + p:  # Retrain
                # XX = X[k - win-p+1:k-p+1, :]
                # J = XNU_B.Jacobian(XX)

                J = JAC[k - win - p + 1:k - p + 1, :]
                break_constant = 0  # -----------------------------------------------2-----------------------------
                if self.ids.break_pretrain_s.active == True:
                    for i in intervals:
                        if k >= i[0] and k < i[1] + win+p:
                            break_constant = 1
                LE_constant = 1  # for right calculation of LE
                for epoch in range(epoch2):
                    if break_constant == 1:  # -------------------------------------2---------------------------------------
                        break

                    yn_temporary[k - win + 1:k + 1] = XNU_B.Y(J, col_W)
                    col_W,dw = BP.LM(yr[k - win + 1:k + 1], yn_temporary[k - win + 1:k + 1], learning_rate, col_W, J)
                    e = yr[k - win + 1:k + 1] - yn[k - win + 1:k + 1]
                    SSE = np.append(SSE, np.dot(e, e))
                self.ids.finished.text = str(
                    'Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
                self.ids.finished.color = [1, 0, 0, 1]
            """For prediction we only need one row of samples as they include all reccurent values"""
            # XX = X[k :k+1, :]  # staci X[k-1,:], takto je temporary.shape = 1,
            # J = XNU_B.Jacobian(XX)
            J = JAC[k:k + 1, :]
            temporary = XNU_B.Y(J, col_W)
            yn[k + p] = temporary[-1]  # PROC k+p-1
            if allow_e == True:# -------------------------------------3---------------------------------------
                if LE_constant == 1:
                    Wm = nn.Wm_operation(Wm, col_W)
                    LE_constant = 0
                self.EA[k, :] = nn.fcnEA(Wm, alfas, OLEs)  # LE # spravne tohle ma byt v if
                self.EAP[k, :] = self.EAP[k - 1, :] + self.EA[k, :]
                #else:
                    #self.EA[k, :] = 0  # LE
                    #self.EAP[k, :] =self.EAP[k - 1, :] +self.EA[k, :]
            # cancel calc.
            if self.cancel == True:# ---------------------------------3-------------------------------------------
                break

        end = time.time() - start
        self.ids.calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.finished.text = str('Finished: ' + str('100' + '%'))
        self.ids.finished.color = [0, 1, 0, 1]

        self.SSE = SSE
        self.e[:L] = yr[:L] - yn[:L]  # ------------------------------------------4----------------------------------
        self.yn = yn
        for k in range(win + p, L):  # Smycka na konci at nezpomaluje vypocet
            self.MAE[k] = np.mean(np.abs(yr[win:k] - yn[win:k]))
            self.RMSE[k] = (np.mean((yr[win:k] - yn[
                                                 win:k]) ** 2)) ** 0.5  # ------------------4----------------------------------------------------------

        return yn
    ### NEW
    def XGD_sample_MLP(self,type, nodes, X, yr, p, win, re_learn, epoch1, epoch2, learning_rate):  # NGD = 0.005, GD = NGD/10

        L = int(yr.shape[0])
        X = nn.input_shape(X)
        yr = nn.output_shape(yr)
        """The dimension of neuron and all matrixes like weights and so on
        depends on number of columns of input vector"""
        X = nn.X_bias(X)
        DIM = nn.input_dim(X[0, :])
        """Creating Neuron object with particular dimension"""

        mlp = nn.MLP(DIM, nodes)
        W = mlp.W
        V = mlp.V

        """Creating Method object(BackPropagation) with particular method"""
        BP = nn.Backpropagation()
        SSE = 0
        yn = np.zeros(L + p)

        #### LE# ---------------------------------1-------------------------------------------
        ole = int(self.ids.OLES.text)
        memory = int(self.ids.memory.text)
        Wm, self.EA, self.EAP, OLEs = nn.Learning_entropy(V, ole, L, p, memory)
        alfas = self.ids.alfas.text
        alfas = np.fromstring(alfas, dtype=float, sep=' ')
        multiply = float(self.ids.multiply.text)
        alfas = alfas / multiply
        allow_e = self.ids.entr_allow.active
        ## MAE RMSE
        self.e = np.zeros(L + p)
        self.MAE = np.zeros(L + p)
        self.RMSE = np.zeros(L + p)
        ### BREAK Retraining
        try:
            intervals = np.fromstring(self.ids.break_intervals.text, dtype=int, sep=' ')
            intervals = intervals.reshape((intervals.shape[0] / 2, 2))
        except:
            intervals = np.array([1, 2])
        LE_constant = 1

            ##### -----------------------------------1-----------------------------------------


        start = time.time()
        for epoch in range(epoch1):
            Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X[:win], yr[p:p + win], win)
            for i in range(win):
                V = BP.NGD_MLP(e[i], learning_rate, Jv[i, :], V)
                # dv = np.dot(np.dot(np.linalg.inv((np.dot(Jv.T, Jv) + 1. / muv * Lv)), Jv.T), e)
                # V = V + dv

                for nod in range(nodes):
                    Jw = Jww[:, nod, :]
                    W[nod, :] = BP.NGD_MLP(e[i], learning_rate, Jw[i, :], W[nod, :])

            SSE = np.append(SSE, np.dot(e, e))
        yn = np.zeros(L + p)
        """Sliding window"""
        for k in range(win + p, L):
            """Re-Learn every x samples, here we relearn every day"""
            if k % (re_learn) == 0 and k >= win + p:  # Retrain
                # XX = X[k - win-p+1:k-p+1, :]
                # J = XNU_B.Jacobian(XX)
                X_ = X[k - win - p + 1:k - p + 1, :]
                #Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X_, yr[k - win + 1:k + 1], win)
                break_constant = 0  # -----------------------------------------------2-----------------------------
                if self.ids.break_pretrain_s.active == True:
                    for i in intervals:
                        if k >= i[0] and k < i[1] + win+p:
                            break_constant = 1
                LE_constant = 1  # for right calculation of LE

                for epoch in range(epoch2):
                    if break_constant == 1:  # -------------------------------------2---------------------------------------
                        break
                    Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X_, yr[k - win + 1:k + 1], win)
                    for i in range(win):
                        if type == 'NGD':
                            V = BP.NGD_MLP(e[i], learning_rate, Jv[i, :], V)
                        if type == 'GD':
                            V = BP.GD_MLP(e[i], learning_rate, Jv[i, :], V)

                        # dv = np.dot(np.dot(np.linalg.inv((np.dot(Jv.T, Jv) + 1. / muv * Lv)), Jv.T), e)
                        # V = V + dv

                        for nod in range(nodes):
                            Jw = Jww[:, nod, :]
                            if type == 'NGD':
                                W[nod, :] = BP.NGD_MLP(e[i], learning_rate, Jw[i, :], W[nod, :])
                            if type == 'GD':
                                W[nod, :] = BP.GD_MLP(e[i], learning_rate, Jw[i, :], W[nod, :])
                    self.ids.finished.text = str(
                        'Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
                    self.ids.finished.color = [1, 0, 0, 1]
                    SSE = np.append(SSE, np.dot(e, e))


            """For prediction we only need one row of samples as they include all reccurent values"""
            Nu = np.dot(W, X[k, :].T)  # n1 x N
            X1 = mlp.phi(Nu)
            yn[k + p] = np.dot(V, X1)
            if allow_e == True:# -------------------------------------3---------------------------------------
                if LE_constant == 1:
                    Wm = nn.Wm_operation(Wm, V)
                    LE_constant = 0
                self.EA[k, :] = nn.fcnEA(Wm, alfas, OLEs)  # LE # spravne tohle ma byt v if
                self.EAP[k, :] = self.EAP[k - 1, :] + self.EA[k, :]
                #else:
                    #self.EA[k, :] = 0  # LE
                    #self.EAP[k, :] =self.EAP[k - 1, :] +self.EA[k, :]
            # cancel calc.
            if self.cancel == True:# ---------------------------------3-------------------------------------------
                break

        end = time.time() - start
        self.ids.calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.finished.text = str('Finished: ' + str('100' + '%'))
        self.ids.finished.color = [0, 1, 0, 1]

        self.SSE = SSE
        self.e[:L] = yr[:L] - yn[:L]  # ------------------------------------------4----------------------------------
        self.yn = yn
        for k in range(win + p, L):  # Smycka na konci at nezpomaluje vypocet
            self.MAE[k] = np.mean(np.abs(yr[win:k] - yn[win:k]))
            self.RMSE[k] = (np.mean((yr[win:k] - yn[
                                                 win:k]) ** 2)) ** 0.5  # ------------------4----------------------------------------------------------

        return yn

    def batch_MLP(self,method, nodes, X, yr, p, win, re_learn, epoch1, epoch2, learning_rate):
        L = int(yr.shape[0])
        X = nn.input_shape(X)
        yr = nn.output_shape(yr)
        """The dimension of neuron and all matrixes like weights and so on
        depends on number of columns of input vector"""
        X = nn.X_bias(X)
        DIM = nn.input_dim(X[0, :])
        """Creating Neuron object with particular dimension"""

        mlp = nn.MLP(DIM, nodes)
        W = mlp.W
        V = mlp.V
        BP = nn.Backpropagation()
        CGD = BP.CGD()

        """Creating Method object(BackPropagation) with particular method"""
        yn = np.zeros(L + p)
        VV = np.zeros((L + p, nodes))
        start = time.time()
        SSE = 0

        #### LE# ---------------------------------1-------------------------------------------
        ole = int(self.ids.OLES.text)
        memory = int(self.ids.memory.text)
        Wm, self.EA, self.EAP, OLEs = nn.Learning_entropy(V, ole, L, p, memory)
        alfas = self.ids.alfas.text
        alfas = np.fromstring(alfas, dtype=float, sep=' ')
        multiply = float(self.ids.multiply.text)
        alfas = alfas / multiply
        allow_e = self.ids.entr_allow.active
        ## MAE RMSE
        self.e = np.zeros(L + p)
        self.MAE = np.zeros(L + p)
        self.RMSE = np.zeros(L + p)
        ### BREAK Retraining
        try:
            intervals = np.fromstring(self.ids.break_intervals.text, dtype=int, sep=' ')
            intervals = intervals.reshape((intervals.shape[0] / 2, 2))
        except:
            intervals = np.array([1, 2])
        LE_constant = 1

            ##### -----------------------------------1-----------------------------------------





        if method == 'LM':
            for i in range(epoch1):
                J, e = mlp.Jacobian(W, V, X[:win, :], yr[p:p + win], win)
                W, V = mlp.W_V_LM(W, V, J, learning_rate, e)
                SSEE = np.dot(e, e)
                SSE = np.append(SSE, SSEE)
        if method == 'CGD':
            J, e = mlp.Jacobian(W, V, X[:win, :], yr[p:p + win], win)  #### MOZNA POTREBUJU Z TOHODLE ZMRDA yn a ne e!!!
            A = CGD.A(J)
            b = CGD.b(yr[p:p + win], J)
            col_W = mlp.pack_WV(W, V)
            re = CGD.re(b, A, col_W)
            pp = re
            for epoch in range(epoch2):
                J, e = mlp.Jacobian(W, V, X[:win, :], yr[p:p + win], win)
                A = CGD.A(J)
                col_W = mlp.pack_WV(W, V)
                col_W, pp, re = CGD.CGD(A, col_W, re, pp)
                W, V = mlp.unpack_WV(col_W)
        yn = np.zeros(L + p)

        """Sliding window"""
        for k in range(win + p, L):
            """Re-Learn every x samples, here we relearn every day"""
            if k % (re_learn) == 0 and k >= win + p:  # Retrain
                J, e = mlp.Jacobian(W, V, X[k - win - p + 1:k - p + 1, :], yr[k - win + 1:k + 1], win)
                A = CGD.A(J)
                b = CGD.b(yr[p:p + win], J)
                col_W = mlp.pack_WV(W, V)
                re = CGD.re(b, A, col_W)
                pp = re

                break_constant = 0  # -----------------------------------------------2-----------------------------
                if self.ids.break_pretrain_s.active == True:
                    for i in intervals:
                        if k >= i[0] and k < i[1] + win+p:
                            break_constant = 1
                LE_constant = 1  # for right calculation of LE

                for i in range(epoch2):
                    if break_constant == 1:  # -------------------------------------2---------------------------------------
                        break
                    if method == 'LM':
                        J, e = mlp.Jacobian(W, V, X[k - win - p + 1:k - p + 1, :], yr[k - win + 1:k + 1], win)
                        W, V = mlp.W_V_LM(W, V, J, learning_rate, e)


                    if method == 'CGD':
                        J, e = mlp.Jacobian(W, V, X[k - win - p + 1:k - p + 1, :], yr[k - win + 1:k + 1], win)
                        A = CGD.A(J)
                        col_W, pp, re = CGD.CGD(A, col_W, re, pp)
                        W, V = mlp.unpack_WV(col_W)


                    SSEE = np.dot(abs(e), abs(e))
                    SSE = np.append(SSE, SSEE)
                self.ids.finished.text = str(
                    'Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
                self.ids.finished.color = [1, 0, 0, 1]
            VV[k, :] = V
            v = np.dot(W, X[k:k + 1, :].T)  # n1 x N
            phi = mlp.phi(v)
            y_temp = mlp.Y(V, phi)
            yn[k + p] = y_temp[-1]
            if allow_e == True:# -------------------------------------3---------------------------------------
                if LE_constant == 1:
                    Wm = nn.Wm_operation(Wm, V)
                    LE_constant = 0
                self.EA[k, :] = nn.fcnEA(Wm, alfas, OLEs)  # LE # spravne tohle ma byt v if
                self.EAP[k, :] = self.EAP[k - 1, :] + self.EA[k, :]
                #else:
                    #self.EA[k, :] = 0  # LE
                    #self.EAP[k, :] =self.EAP[k - 1, :] +self.EA[k, :]
            # cancel calc.
            if self.cancel == True:# ---------------------------------3-------------------------------------------
                break

        end = time.time() - start
        self.ids.calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.finished.text = str('Finished: ' + str('100' + '%'))
        self.ids.finished.color = [0, 1, 0, 1]
        self.SSE = SSE
        self.e[:L] = yr[:L] - yn[:L]  # ------------------------------------------4----------------------------------
        self.yn = yn
        for k in range(win + p, L):  # Smycka na konci at nezpomaluje vypocet
            self.MAE[k] = np.mean(np.abs(yr[win:k] - yn[win:k]))
            self.RMSE[k] = (np.mean((yr[win:k] - yn[
                                                 win:k]) ** 2)) ** 0.5  # ------------------4----------------------------------------------------------
        return yn

    def XGD_sample_HONU(self,method,type, X, yr, p, win, re_learn, epoch1, epoch2, learning_rate):
        start = time.time()
        L = int(yr.shape[0])
        X = nn.input_shape(X)
        yr = nn.output_shape(yr)
        """The dimension of neuron and all matrixes like weights and so on
        depends on number of columns of input vector"""
        DIM = nn.input_dim(X[0, :])
        """Creating Neuron object with particular dimension"""
        if type == 'CNU':
            XNU_S = nn.HONU().CNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_S.col_W_from_W(XNU_S.W_CNU_init)
        if type == 'QNU':
            XNU_S = nn.HONU().QNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_S.col_W_from_W(XNU_S.W_QNU_init)
        if type == 'LNU':
            XNU_S = nn.HONU().LNU_Batch(DIM)  # +number_of_rec)
            col_W = XNU_S.col_W_from_W(XNU_S.W_LNU_init)
        """Creating Method object(BackPropagation) with particular method"""
        BP = nn.Backpropagation()
        yn = np.zeros(L + p)

        e = np.zeros(L + p)

        self.ids.jacobian_calcul.text = 'being calculated'
        self.ids.jacobian_calcul.color = [1, 0, 0, 1]
        start = time.time()

        JAC = XNU_S.Jacobian(X[:, :])
        end = time.time()-start

        self.ids.jacobian_calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.jacobian_calcul.text = 'calculation finished'
        self.ids.jacobian_calcul.color = [0, 1, 0, 1]

        #### LE# ---------------------------------1-------------------------------------------
        ole = int(self.ids.OLES.text)
        memory = int(self.ids.memory.text)
        Wm, self.EA, self.EAP, OLEs = nn.Learning_entropy(col_W, ole, L, p, memory)
        alfas = self.ids.alfas.text
        alfas = np.fromstring(alfas, dtype=float, sep=' ')
        multiply = float(self.ids.multiply.text)
        alfas = alfas / multiply
        allow_e = self.ids.entr_allow.active
        ## MAE RMSE
        self.e = np.zeros(L + p)
        self.MAE = np.zeros(L + p)
        self.RMSE = np.zeros(L + p)
        ### BREAK Retraining
        try:
            intervals = np.fromstring(self.ids.break_intervals.text, dtype=int, sep=' ')
            intervals = intervals.reshape((intervals.shape[0] / 2, 2))
        except:
            intervals = np.array([1, 2])
        LE_constant = 1

            ##### -----------------------------------1-----------------------------------------


        J = JAC[:win, :]
        SSE = 0
        for epoch in range(epoch1):
            for k in range(win):
                j = J[k, :]
                yn[k + p] = XNU_S.Y(j, col_W)
                e[k + p] = yr[k + p] - yn[k + p]
                # dw = learning_rate / (1 + np.sum(j ** 2)) * e[k + p] * j
                # col_W = col_W + dw
                if method == 'NGD':
                    col_W = BP.NGD(yr[k + p], yn[k + p], learning_rate, j, col_W)
                if method =='GD':
                    col_W = BP.GD(yr[k + p], yn[k + p], learning_rate, j, col_W)

            SSE = np.append(SSE, np.sum((e[p:p + win]) ** 2))


        # for i in range(epoch1):
        #    yn[p:p + win] = XNU_B.Y(J, col_W)
        #    col_W = BP.LM(yr[p:p + win], yn[p:p + win], learning_rate, col_W,
        #                  J)  # y_target,y_neuron, learning rate, col_W, Jacobian(col_X)
        yn = np.zeros(L + p)

        """Sliding window"""
        for k in range(win + p, L):
            """Re-Learn every x samples, here we relearn every day"""
            if k % (re_learn) == 0 and k >= win + p:  # Retrain
                # XX = X[k - win-p+1:k-p+1, :]
                # J = XNU_B.Jacobian(XX)
                J = JAC[k - win - p + 1:k - p + 1, :]
                e = np.zeros(win)
                break_constant = 0  # -----------------------------------------------2-----------------------------
                if self.ids.break_pretrain_s.active == True:
                    for i in intervals:
                        if k >= i[0] and k < i[1] + win+p:
                            break_constant = 1
                LE_constant = 1  # for right calculation of LE

                for epoch in range(epoch2):
                    if break_constant == 1:  # -------------------------------------2---------------------------------------
                        break
                    for i in range(win):
                        j = J[i, :]
                        # yn[k - win + 1+i] = XNU_S.Y(j, col_W)
                        yn_temp = XNU_S.Y(j, col_W)
                        e[i] = yr[k - win + 1 + i] - yn_temp

                        # dw = learning_rate / (1 + np.sum(j ** 2)) * e[i] * j
                        # col_W = col_W + dw
                        if method == 'NGD':
                            col_W = BP.NGD(yr[k - win + 1 + i], yn_temp, learning_rate, j, col_W)
                        if method == 'GD':
                            col_W = BP.GD(yr[k - win + 1 + i], yn_temp, learning_rate, j, col_W)

                    SSE = np.append(SSE, np.dot(e,e))
                    # SSE[epoch] = sum((e) ** 2)
                self.ids.finished.text = str(
                    'Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
                self.ids.finished.color = [1, 0, 0, 1]
            """For prediction we only need one row of samples as they include all reccurent values"""
            # XX = X[k :k+1, :]  # staci X[k-1,:], takto je temporary.shape = 1,
            # J = XNU_B.Jacobian(XX)
            j = JAC[k, :]
            temporary = XNU_S.Y(j, col_W)
            yn[k + p] = temporary  # PROC k+p-1
            if allow_e == True:# -------------------------------------3---------------------------------------
                if LE_constant == 1:
                    Wm = nn.Wm_operation(Wm, col_W)
                    LE_constant = 0
                self.EA[k, :] = nn.fcnEA(Wm, alfas, OLEs)  # LE # spravne tohle ma byt v if
                self.EAP[k, :] = self.EAP[k - 1, :] + self.EA[k, :]
                #else:
                    #self.EA[k, :] = 0  # LE
                    #self.EAP[k, :] =self.EAP[k - 1, :] +self.EA[k, :]
            # cancel calc.
            if self.cancel == True:# ---------------------------------3-------------------------------------------
                break

        end = time.time() - start
        self.ids.calcul_time.text = str(round(end, 2)) + ' s'
        self.ids.finished.text = str('Finished: ' + str('100' + '%'))
        self.ids.finished.color = [0, 1, 0, 1]
        self.e[:L] = yr[:L] - yn[:L]  # ------------------------------------------4----------------------------------
        self.yn = yn
        for k in range(win + p, L):  # Smycka na konci at nezpomaluje vypocet
            self.MAE[k] = np.mean(np.abs(yr[win:k] - yn[win:k]))
            self.RMSE[k] = (np.mean((yr[win:k] - yn[
                                                 win:k]) ** 2)) ** 0.5  # ------------------4----------------------------------------------------------
        return yn

    def coarse_grain_data(self,data):
        radius = self.ids.radius.text
        try:
            radius = int(radius)
            y_cg = nn.coarse_grain(data,radius)

        except:
            pass
        return y_cg

    def set_recommended_intervals(self): # Generovani intervalu pro preruseni
        path = 'evaluation\\'
        with open(path + "signal_intervals.txt", "r") as myfile:
            data = myfile.read().replace('\n', '\n')
        array = np.fromstring(data, dtype=int, sep=' ')
        if self.ids.tolerance.text== '':
            self.ids.tolerance.text='0 0'
        tolerance = np.fromstring(self.ids.tolerance.text, dtype=int, sep=' ')
        for i in range(array.shape[0]):
            if i>1:
                if i%2==0:
                    array[i]-=tolerance[0]
                else:
                    array[i]+=tolerance[1]
        data = ''
        for i in array:
            data += str(int(i)) + ' '

        self.ids.break_intervals.text = data

    """
    ---------------Graphic logic--------------
    """
    def cg(self):  # inicializovani mnoha konstant po vytvoreni Grafu po kliknuti na Graph
        if self.constant == 0:
            path = 'free mode code\\'
            print('Created')
            self.create_graph()
            self.constant +=1
            self.ids.refresh_button.disabled = False
            self.ids.run_button.disabled = False
            #self.ids.show_button.disabled = False
            self.ids.code_button.disabled = False
            self.ids.show_code_button.disabled = False
            #self.ids.result_code_button.disabled = False
            self.ids.prediction_button.disabled = False
            self.ids.data_button.disabled = False
            with open(path+"example_code_1.txt", "r") as myfile:
                data = myfile.read().replace('\n', '\n')
            self.ids.code_it.text = data
            try:
                self.ids.data_type.max = self.X.shape[1]-1

            except:
                self.ids.data_type.max = 1

    def load_code(self): # Narhani kodu do free modu
        path = 'free mode code\\'
        try:
            with open(path+self.ids.code_text.text, "r") as myfile:
                data = myfile.read().replace('\n', '\n')
            self.ids.code_it.text = data
        except:
            pass
    def save_as(self): # ulozeni kodu free modu
        path = 'free mode code\\'
        with open(path+self.ids.code_text.text, "w") as text_file:
            text_file.write(self.ids.code_it.text)

    def create_graph(self): # vytvoreni grafu
        #fig,self.ax = plt.figure()#(facecolor=[0, 0, 0, 1])  # Facecolor
        fig, self.ax = plt.subplots()  # (facecolor=[0, 0, 0, 1])  # Facecolor
        #self.ax = fig.add_subplot(211)
        #self.ax2 = fig.add_subplot(212)
        self.canvas = FigureCanvasKivyAgg(figure=fig)
        graph_window = self.ids.graph_window
        graph_window.add_widget(self.canvas)
        self.LIST_OF_GRAPHS.append(self.canvas) # neni uplne nutny
        self.createe_graph()

    def add_action_bar(self): # zobrazeni a mazani pomocne listy pro matplotlib
        graph_window = self.ids.graph_window
        try:
            if graph_window.width >= 500 and len(self.LIST_OF_ACTION_BARS)==0:

                nav1 = NavigationToolbar2Kivy(self.canvas)
                navigation = nav1.actionbar
                graph_window.add_widget(navigation)
                self.LIST_OF_ACTION_BARS.append(navigation)
            if graph_window.width < 500 and len(self.LIST_OF_ACTION_BARS)>0:
                graph_window = self.ids.graph_window
                for i in self.LIST_OF_ACTION_BARS:
                    graph_window.remove_widget(i)
                self.LIST_OF_ACTION_BARS = []
        except:
            pass



    def delete_graph(self):
        graph_window = self.ids.graph_window
        for i in self.LIST_OF_GRAPHS:
            graph_window.remove_widget(i)

    def createe_graph(self):
        self.ax.cla()
        self.graph, = self.ax.plot([],[],'k')
        self.graph2, = self.ax.plot([],[],'g')
        self.graph3, = self.ax.plot([], [], 'r', linewidth=1, linestyle='--')
        self.add_action_bar()






    def save_data_to_show_entropy(self):
        threading.Thread(target = self.save_data_to_show_ent).start()

    def save_yn(self):
        np.savetxt('yn.txt', self.yn*self.normalize_data)

    def save_data_to_show_ent(self): # ulozeni promennych do TXT
        path = 'evaluation\\'
        np.savetxt(path+'yr.txt',self.yr*self.normalize_data)
        np.savetxt(path+'yn.txt', self.yn*self.normalize_data)
        np.savetxt(path+'e.txt', self.e*self.normalize_data)
        np.savetxt(path+'EA.txt', self.EA)
        np.savetxt(path+'EAP.txt', self.EAP)
        #np.savetxt('error_settings.txt', self.error_settings)
        os.system(path+"learn_entr.py")
    def save_data_to_show_error(self):
        threading.Thread(target = self.save_data_to_show_err).start()
    def save_data_to_show_err(self): # ulozeni promennych do TXT
        path = 'evaluation\\'
        e_ari = nn.arithmetic_error(self.e*self.normalize_data, int(self.ids.error_memory.text))
        if self.ids.val_def.active == True:
            error_percentage = nn.ratio_yr_to_ari_er(float(self.ids.value_defined.text), e_ari)*100
        if self.ids.val_aver.active == True:
            average_value = np.sum(self.yr*self.normalize_data) / self.yr.shape[0]
            error_percentage = nn.ratio_yr_to_ari_er(average_value, e_ari)*100
        np.savetxt(path+'error_percentage.txt',error_percentage)
        if self.ids.eval_.active == True:
            np.savetxt(path+'error_settings.txt', np.array([1,float(self.ids.error_percentage_border.text)]))
        if self.ids.eval_.active == False:
            np.savetxt(path+'error_settings.txt', np.array([0,float(self.ids.error_percentage_border.text)]))
        np.savetxt(path+'yr.txt', self.yr * self.normalize_data)
        np.savetxt(path+'yr.txt',self.yr*self.normalize_data)
        np.savetxt(path+'yn.txt', self.yn*self.normalize_data)
        np.savetxt(path+'e.txt', self.e*self.normalize_data)
        np.savetxt(path+'EA.txt', self.EA)
        np.savetxt(path+'EAP.txt', self.EAP)
        np.savetxt(path+'RMSE.txt', self.RMSE)
        np.savetxt(path+'MAE.txt', self.MAE)

        rmse_checkbox = self.ids.rmse_mae_check.active
        err_p_checkbox=self.ids.error_p_check.active
        np.savetxt(path+'what_to_plot.txt',np.array([int(rmse_checkbox),int(err_p_checkbox)]))
        #np.savetxt('error_settings.txt', self.error_settings)
        os.system(path+"error_plot.py")

    def set_neural_data(self): # Zobrazeni promenych do grafu
        self.ids.graph_accordion.collapse = False
        self.ids.info.collapse = True
        X_xaxis = np.arange(self.X_last.shape[0])
        Y_yaxis = np.arange(self.yn.shape[0])
        E_xaxis = np.arange(self.e.shape[0])
        X = self.X_last*self.normalize_data
        Y = self.yn * self.normalize_data
        self.ax.cla()
        self.graph, = self.ax.plot(X_xaxis, X, 'k')
        self.graph2, = self.ax.plot(Y_yaxis, Y, 'g')
        self.graph3, = self.ax.plot(E_xaxis, abs(self.e * self.normalize_data), 'r', linewidth=1, linestyle='--')
        y1max = np.amax(Y)
        y2max = np.amax(X)
        y3max = np.amax(abs(self.e * self.normalize_data))
        ymax = np.amax([y1max, y2max, y3max])
        y1min = np.amin(Y)
        y2min = np.amin(X)
        y3min = np.amin(abs(self.e * self.normalize_data))
        ymin = np.amin([y1min, y2min, y3min])
        plt.xlim(Y_yaxis[0], Y_yaxis[-1])
        plt.ylim(ymin, ymax)
        #self.graph.set_xdata(X_xaxis)
        #self.graph.set_ydata(X)
        #self.graph2.set_xdata(Y_yaxis)
        #self.graph2.set_ydata(Y)
        #self.graph3.set_xdata(E_xaxis)
        #self.graph3.set_ydata(abs(self.e*self.normalize_data))
        #self.ax.set_ylim(0, np.amax(X)*1.3)
        #self.ax.set_xlim(0,self.yn.shape[0])
        self.canvas.draw()


    def set_neural_data_code(self): # Zobrazeni promenych do grafu
        self.ids.graph_accordion.collapse = False
        self.ids.info.collapse = True
        X_xaxis = np.arange(self.code_yr.shape[0])
        Y_yaxis = np.arange(self.yn.shape[0])
        X = self.code_yr * self.normalize_data
        Y = self.yn * self.normalize_data
        E_xaxis = np.arange(self.e.shape[0])
        self.ax.cla()
        self.graph, = self.ax.plot(X_xaxis, X, 'k')
        self.graph2, = self.ax.plot(Y_yaxis,Y, 'g')
        self.graph3, = self.ax.plot(E_xaxis,abs(self.e * self.normalize_data) , 'r', linewidth=1, linestyle='--')

        y1max = np.amax(Y)
        y2max = np.amax(X)
        y3max = np.amax(abs(self.e * self.normalize_data))
        ymax = np.amax([y1max,y2max,y3max])
        y1min = np.amin(Y)
        y2min = np.amin(X)
        y3min = np.amin(abs(self.e * self.normalize_data))
        ymin = np.amin([y1min, y2min, y3min])
        plt.xlim(Y_yaxis[0],Y_yaxis[-1])
        plt.ylim(ymin,ymax)


        #self.graph.set_xdata(X_xaxis)
        #self.graph.set_ydata(X)
        #self.graph2.set_xdata(Y_yaxis)
        #self.graph2.set_ydata(Y)
        #self.graph3.set_xdata(E_xaxis)
        #self.graph3.set_ydata(abs(self.e * self.normalize_data))
        #self.ax.set_ylim(np.amin(X) - np.abs(np.amin(X))*0.3 , np.amax(X) * 1.3)
        #self.ax.set_xlim(-5, self.yn.shape[0]*1.1)
        self.canvas.draw()

    def set_data(self): # Zobrazeni promenych do grafu
        # self.ax.cla()

        X_data = np.arange(self.X.shape[0])
        # try:
        data__type = int(self.ids.data_type.value)
        #    if data__type > 10 or data__type < 0:
        #        data__type == 1
        # except:
        #    data__type= 1
        try:
            sd1 = int(self.ids.sda.text) * self.sampling
            sd2 = int(self.ids.sdb.text) * self.sampling
        except:
            sd1 = 3 * self.sampling
            sd2 = (3 + 21) * self.sampling

        if sd2 < (sd1 + 14 * self.sampling):
            sd2 = sd1 + 21 * self.sampling
        Y_data = self.X[sd1:sd2, data__type]
        try:
            if self.ids.cgrain_on_off.active == True and int(self.ids.radius.text) > 0:
                Y_data = self.coarse_grain_data(Y_data)
        except:
            print('cislo musi byt cele a kladne')

        self.ax.cla()
        self.graph, = self.ax.plot(X_data[sd1:sd2], Y_data, 'k')
        plt.xlim(X_data[sd1], X_data[sd2-1])
        plt.ylim(np.amin(Y_data), np.amax(Y_data))

        #self.graph.set_xdata(X_data[sd1:sd2])
        #self.graph.set_ydata(Y_data)
        #self.ax.set_ylim(0, np.amax(Y_data) * 1.3)
        #self.ax.set_xlim(sd1 + 10, sd2 + 10)
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 10000))
        # self.ax.set_title('Spread (XBT/EUR)', color=[1, 1, 1, 1], fontsize=20)
        # self.ax.set_xlabel('History (XBT/EUR)', color=[1, 1, 1, 1], fontsize=15)
        # self.ax.set_ylabel('Asks/Bids (XBT/EUR)', color=[1, 1, 1, 1], fontsize=15)
        self.canvas.draw()

    def set_data_code(self): # Zobrazeni promenych do grafu
        self.ids.graph_accordion.collapse = False
        self.ids.info.collapse = True
        self.ax.cla()
        self.graph, = self.ax.plot([], [], 'k')
        #self.graph2, = self.ax.plot([], [], 'g')
        #self.graph3, = self.ax.plot([], [], 'r', linewidth=1, linestyle='--')

        import numpy as np
        # self.ax.cla()

        X_data = self.code_x_axis
        Y_data = self.code_yr * self.normalize_data
        self.ax.cla()
        self.graph, = self.ax.plot(X_data, Y_data, 'k')
        plt.xlim(X_data[0], X_data[-1])
        plt.ylim(np.amin(Y_data), np.amax(Y_data))

        #self.graph.set_xdata(X_data)
        #self.graph.set_ydata(Y_data)
        #self.ax.set_ylim(np.amin(Y_data) - np.abs(np.amin(Y_data)) * 0.3, np.abs(np.amax(Y_data)) * 1.3)
        #elf.ax.set_xlim(0, X_data[-1])

        self.canvas.draw()

    def fcnEA(self,Wm, alphas, OLEs):  # Wm ... recent window of weights including the very last weight updates
        OLEs = OLEs.reshape(-1)
        nw = Wm.shape[1]
        nalpha = len(alphas)
        ea = np.zeros(len(OLEs))
        i = 0
        for ole in range(max(OLEs) + 1):
            if ole == OLEs[i]:  # assures the corresponding difference of Wm
                absdw = np.abs(Wm[-1, :])  # very last updated weights
                meanabsdw = np.mean(abs(Wm[0:Wm.shape[0] - 1, :]), 0)
                Nalpha = 0
                for alpha in alphas:
                    Nalpha += sum(absdw > alpha * meanabsdw)
                ea[i] = float(Nalpha) / (nw * nalpha)
                i += 1
            Wm = Wm[1:, :] - Wm[0:(np.shape(Wm)[0] - 1), :]  # difference Wm
        return (ea)

class guiApp(App): # Konec tridy GUI
    pass

if __name__ == '__main__':
    guiApp().run()
