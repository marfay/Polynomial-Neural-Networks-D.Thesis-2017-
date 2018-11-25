import numpy as np
import neural_network as nn
import time
#######--------------------------------PREDIKCE NEURONOVYCH SITI---------------------------------------------------------

"""
Na konci jsou implementace, staci odkomentovat examples....
"""



"""
Okomentuji pouze unikaty. Pokud se funkce budou opakovat, vynecham komentar.
v dolni casti jsou sablony pro pouziti funkci

"""
def CGD_batch(type,X,yr,p,win,re_learn,epoch1,epoch2):
    '''

    :param type: 'CNU','QNU','LNU'
    :param X: vstupni matice
    :param yr: namerene hodnoty
    :param p: predikce
    :param win: uciciho sirka okna
    :param re_learn:  po kolika vzorcich preucit
    :param epoch1: pocet uceni pro predtrenovani
    :param epoch2: pocet epoch uceni pro pretrenovavani
    :return:
    '''

    start=time.time() # Zacatek mereni
    L = int(yr.shape[0]) # sirka vektoru namerenych hodnot - musi byt stejna jako pro X
    X = nn.input_shape(X) # Uprava vektoru/matice viz neural_network.py
    yr = nn.output_shape(yr) # Uprava vektoru viz neural_network.py
    """The dimension of neuron and all matrixes like weights and so on
    depends on number of columns of input vector"""
    DIM = nn.input_dim(X[0, :]) # X je vstupni matice, proto X[0, :] je vstupni vektor do neuronu. Touto funkci ziskam sirku vs. vektoru
    """Creating Neuron object with particular dimension"""
    if type == 'CNU':
        XNU_B = nn.HONU().CNU_Batch(DIM)   # vznik objektu neuronu daneho typu
        col_W = XNU_B.col_W_from_W(XNU_B.W_CNU_init) # ziskani dlouheho plocheho vektoru vah z matice vah
    if type == 'QNU':
        XNU_B = nn.HONU().QNU_Batch(DIM)
        col_W = XNU_B.col_W_from_W(XNU_B.W_QNU_init)
    if type == 'LNU':
        XNU_B = nn.HONU().LNU_Batch(DIM)
        col_W = XNU_B.col_W_from_W(XNU_B.W_LNU_init)
    """Creating Method object(BackPropagation) with particular method"""
    BP = nn.Backpropagation() # vytvoreni objektu ucicich algorimu BP
    CGD = BP.CGD() # Vytvoreni objektu z BP pro Conjugate Gradient Descent
    yn = np.zeros(L + p) # inicializace vektoru neuronoveho vystupu
    JAC = XNU_B.Jacobian(X[:,:]) # Vytvoreni Jakobianu z cele vstupni matice - kvuli kratsimu vypoctu
    #J = XNU_B.Jacobian(X[:win,:])
    J = JAC[:win,:] # J - kratkodoby jakobian - pro predtrenovani
    A = CGD.A(J) # Symetricka ctvercova matice pro CGD - poc. promenne
    b = CGD.b(yr[p:p+win], J) # poc. promena pro CGD
    re = CGD.re(b, A, col_W)  # poc. promena pro CGD
    pp = re # poc. promena pro CGD


    Wm,EA,EAP,OLEs = nn.Learning_entropy(col_W,4,L,p,15) #
    alfas = np.array([10,  5,  3,  1,   0.5,0.01])*10

    # iterace - velmi rychly vypocet, neni treba pocitat vystup neuronu
    for i in range(epoch1):
        col_W, pp, re = CGD.CGD(A, col_W, re, pp)  # y_target,y_neuron, learning rate, col_W, Jacobian(col_X)
    yn[p:p + win] = XNU_B.Y(J, col_W)

    """Sliding window- metoda predikce klouzaveho okna zacina zde"""
    k=0
    con =1
    for k in range(win+p,L): # konec okna klouze od win+p po L, tedy posledni vypoctena hodnota je L+p
        """Re-Learn every x samples"""
        if k%(re_learn) == 0 and k>=win+p+1: # Podminka, kdy pretrenovat - k%x = modulo,
            #XX = X[k-win-p+1:k-p+1,:]
            #J = XNU_B.Jacobian(XX)
            J = JAC[k-win-p+1:k-p+1,:] # kratkodoby jakobian
            A = CGD.A(J) # # poc. promena pro CGD
            b = CGD.b(yr[k-win+1:k+1], J) # poc. promena pro CGD
            re = CGD.re(b, A, col_W) # poc. promena pro CGD
            pp = re # poc. promena pro CGD

            print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)),2)) + '%')
            con = 1
            # Iterace CGD
            for epoch in range(epoch2):
                col_W, pp, re = CGD.CGD(A, col_W, re, pp)

        """For prediction we only need one row of samples as they include all reccurent values"""
        #XX = X[k:k+1,:] # staci X[k-1,:], takto je temporary.shape = 1,
        # Jde nam o posledni hodnotu, tedy k+p, proto staci udelat kratkodoby jakobian posledniho radku
        J = JAC[k:k+1,:]
        #J = XNU_B.Jacobian(XX)
        temporary = XNU_B.Y(J, col_W) # vystup je cislo
        yn[k+p] = temporary[-1] # zbytecne v tomto pripade - duvod je ten ze je moznost brat treba poslednich 5 hodnot z vypocitaneho vektoru, tady se y pocita kazdy krok(k). Takze pokud krok bude 5, tak by bylo dobre brat 5 poslednich hodnot z vypocitaneho vektory y

        if con == 1: # con = 1 po pretrenovani
            # operace pro vypocet entropie - zpomaluje ale vypocet neuronove site
            Wm = nn.Wm_operation(Wm,col_W)
            con = 0
            EA[k, :] = nn.fcnEA(Wm, alfas, OLEs)  # LE
            EAP[k, :] = EAP[k - 1, :] + EA[k, :]
        else:
            EA[k, :] = 0  # LE
            EAP[k, :] = EAP[k - 1, :] + EA[k, :]

    return yn,EA,EAP



def LM_batch(type,X,yr,p,win,re_learn,epoch1,epoch2,learning_rate):
    start=time.time()
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
    BP = nn.Backpropagation()
    yn = np.zeros(L + p)
    yn_temporary = np.zeros(L + p)

    JAC = XNU_B.Jacobian(X[:, :])
    #J = XNU_B.Jacobian(X[:win,:])
    J = JAC[:win,:]

    print(col_W.shape,J.shape)
    for i in range(epoch1):
        yn[p:p + win] = XNU_B.Y(J, col_W)
        col_W,dw = BP.LM(yr[p:p + win], yn[p:p + win], learning_rate, col_W,
                      J)  # y_target,y_neuron, learning rate, col_W, Jacobian(col_X)


    """Sliding window"""
    for k in range(win+p, L ):
        """Re-Learn every x samples, here we relearn every day"""
        if k % (re_learn) == 0 and k >=win+p:  # Retrain
            J = JAC[k - win - p + 1:k - p + 1, :]
            print(col_W.shape, J.shape)
            for epoch in range(epoch2):
                yn_temporary[k - win+1:k+1] = XNU_B.Y(J, col_W) # u LM je potreba spocitat i vystup z neuronu - proto je algoritmus pomaly oproti CGD
                col_W,dw = BP.LM(yr[k - win+1:k+1 ], yn_temporary[k - win+1 :k+1], learning_rate, col_W, J)
            print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)),2)) + '%')
        """For prediction we only need one row of samples as they include all reccurent values"""
        #XX = X[k :k+1, :]  # staci X[k-1,:], takto je temporary.shape = 1,
        #J = XNU_B.Jacobian(XX)
        J = JAC[k:k + 1, :]
        temporary = XNU_B.Y(J, col_W)
        yn[k + p ] = temporary[-1]
    return yn

def NGD_sample_HONU(type,X,yr,p,win,re_learn,epoch1,epoch2,learning_rate):
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

    JAC = XNU_S.Jacobian(X[:, :])
    # J = XNU_B.Jacobian(X[:win,:])
    J = JAC[:win, :]
    SSE = 0
    for epoch in range(epoch1): # Epoch trenovani
        for k in range(win): # Klouzajici okno
            j = J[k,:]
            yn[k+p] = XNU_S.Y(j,col_W)
            e[k+p] = yr[k + p] - yn[k + p]
            col_W = BP.NGD(yr[k+p], yn[k+p], learning_rate, j, col_W)

        SSE = np.append(SSE,np.sum((e[p:p+win]) ** 2))

    #for i in range(epoch1):
    #    yn[p:p + win] = XNU_B.Y(J, col_W)
    #    col_W = BP.LM(yr[p:p + win], yn[p:p + win], learning_rate, col_W,
    #                  J)  # y_target,y_neuron, learning rate, col_W, Jacobian(col_X)

    """Sliding window"""
    for k in range(win + p, L):
        """Re-Learn every x samples, here we relearn every day"""
        if k % (re_learn) == 0 and k >= win + p:  # Retrain
            # XX = X[k - win-p+1:k-p+1, :]
            # J = XNU_B.Jacobian(XX)
            J = JAC[k - win - p + 1:k - p + 1, :]
            e = np.zeros(win)
            for epoch in range(epoch2):
                for i in range(win):# klouzajici okno
                    j = J[i, :]
                    yn_temp = XNU_S.Y(j, col_W)
                    e[i] = yr[k - win + 1+i] - yn_temp
                    col_W = BP.NGD(yr[k - win + 1+i],yn_temp,learning_rate,j,col_W)
                SSE = np.append(SSE, np.sum((e** 2)))
                #SSE[epoch] = sum((e) ** 2)

            print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
        """For prediction we only need one row of samples as they include all reccurent values"""
        j = JAC[k, :]
        temporary = XNU_S.Y(j, col_W)
        yn[k + p] = temporary
    return yn,SSE


def NGD_sample_MLP(nodes,X,yr,p,win,re_learn,epoch1,epoch2,learning_rate):
    L = int(yr.shape[0])
    X = nn.input_shape(X)
    yr = nn.output_shape(yr)
    """The dimension of neuron and all matrixes like weights and so on
    depends on number of columns of input vector"""
    #POZOR u mlp se pridava bias manualne -> Nikoliv vsak v aplikaci
    X = nn.X_bias(X)
    DIM = nn.input_dim(X[0, :]) # rozmer vstupni matice i s Biasem

    """Creating Neuron object with particular dimension"""
    mlp = nn.MLP(DIM, nodes) # nodes = pocet neuronu ve skryte vrstve
    W = mlp.W
    V = mlp.V

    yn = np.zeros(L + p)
    e = np.zeros(win)
    print(X.shape)
    for epoch in range(epoch1):
        for k in range(win): # klouzajici okno
            x = X[k,:]  # vstupni vektor ze vstupni matice
            nu = np.dot(W, x) # agregace
            x1 = mlp.phi(nu)  # 1st layer output - vystup ze skryte vrstvy
            yn[k + p] = np.dot(V , x1) # vystup ze site
            e[k + p] = yr[k + p] - yn[k + p] # chyba
            # hidden layer updates
            for i in range(nodes): # uceni jednotlivych neuronu skryte vrstvy
                dyndWi = V[i] * mlp.dphidv(nu[i]) * x
                dWi = learning_rate / (1 + np.sum(x ** 2)) * e[k + p ] * dyndWi
                W[i, :] = W[i, :] + dWi

            # output layer updates
            # uceni vystupniho neuronu
            dyndv = x1
            dv = learning_rate * e[k + p] * dyndv
            V = V + dv

        #SSE[epoch] = sum((e[ny + p - 1:] * 3 * self.stdy) ** 2)

    """Sliding window"""
    for k in range(win + p, L):
        """Re-Learn every x samples, here we relearn every day"""
        if k % (re_learn) == 0 and k >= win + p:  # Retrain
            X_= X[k - win - p + 1:k - p + 1, :] # kratkodoba vstupni matice pro dany usek pretrenovani
            e= np.zeros(win)
            for epoch in range(epoch2):
                for ii in range(win): # klouzajici okno - stejny princip jako pri predtrenovani
                    x = X_[ii, :]
                    nu = np.dot(W, x)
                    x1 = mlp.phi(nu)  # 1st layer output
                    yn[k -win +1+ii] = np.sum(V * x1)
                    yn_temp = np.sum(V * x1)
                    #e[k -win +1+ii] = yr[k -win +1+ii] - yn[k -win +1+ii]
                    e[ii] = yr[k - win + 1 + ii] - yn_temp
                    # hidden layer updates
                    for i in range(nodes):
                        dyndWi = V[i] * mlp.dphidv(nu[i]) * x

                        #dWi = learning_rate * e_temp  * dyndWi
                        dWi = learning_rate/(1+np.sum(x**2))*e[ii]*dyndWi
                        W[i, :] = W[i, :] + dWi
                    dyndv = x1

                    dv = learning_rate * e[ii]* dyndv
                    V = V + dv
            print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')



        """For prediction we only need one row of samples as they include all reccurent values"""
        # XX = X[k :k+1, :]  # staci X[k-1,:], takto je temporary.shape = 1,
        # J = XNU_B.Jacobian(XX)
        x = X[k, :]

        nu = np.dot(W, x)
        x1 = mlp.phi(nu)  # 1st layer output
        yn[k + p] = np.sum(V * x1)
        #try:
        #    print(yr[k+p] - yn[k+p])
        #except:
        #    pass
    return yn


 #   def GD_sample_MLP(nodes,X,yr,p,win,re_learn,epoch1,epoch2,learning_rate):
 #
 #       L = int(yr.shape[0])
 #       X = nn.input_shape(X)
 #       yr = nn.output_shape(yr)
 #       """The dimension of neuron and all matrixes like weights and so on
 #       depends on number of columns of input vector"""
 #       X = nn.X_bias(X)
 #       DIM = nn.input_dim(X[0, :])
 #      """Creating Neuron object with particular dimension"""
 #
 #       mlp = nn.MLP(DIM, nodes)
 #       W = mlp.W
 #       V = mlp.V
 #
 #       """Creating Method object(BackPropagation) with particular method"""
 #       BP = nn.Backpropagation()
 #       SSE = 0
 #       yn = np.zeros(L + p)
 #
 #       e = np.zeros(win)
 #       SSE = 0
 #       for epoch in range(epoch1):
 #           Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X[:win], yr[p:p + win], win)
 #           for i in range(win):
 #               V = BP.NGD_MLP(e[i], learning_rate, Jv[i, :], V)
 #               # dv = np.dot(np.dot(np.linalg.inv((np.dot(Jv.T, Jv) + 1. / muv * Lv)), Jv.T), e)
 #               # V = V + dv
  #
  #              for nod in range(nodes):
  #                  Jw = Jww[:, nod, :]
  #                  W[nod, :] = BP.NGD_MLP(e[i], learning_rate, Jw[i, :], W[nod, :])
  #
  #          SSE= np.append(SSE,np.dot(e, e))
  #
  #      """Sliding window"""
  #      for k in range(win + p, L):
  #          """Re-Learn every x samples, here we relearn every day"""
  #         if k % (re_learn) == 0 and k >= win + p:  # Retrain
  #             # XX = X[k - win-p+1:k-p+1, :]
  #             # J = XNU_B.Jacobian(XX)
  #              X_= X[k - win - p + 1:k - p + 1, :]
  #              Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X_, yr[k - win+1:k+1], win)
  #              for epoch in range(epoch2):
  #                 Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X_, yr[k - win + 1:k + 1], win)
  #                  for i in range(win):
  #                     V = BP.NGD_MLP(e[i], learning_rate, Jv[i, :], V)
  #                      # dv = np.dot(np.dot(np.linalg.inv((np.dot(Jv.T, Jv) + 1. / muv * Lv)), Jv.T), e)
                        # V = V + dv
 #
  #                      for nod in range(nodes):
  #                          Jw = Jww[:, nod, :]
  #                          W[nod, :] = BP.NGD_MLP(e[i], learning_rate, Jw[i, :], W[nod, :])
  #
  #                  SSE = np.append(SSE, np.dot(e, e))
  #
  #              print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
  #
  #
   #
   #         """For prediction we only need one row of samples as they include all reccurent values"""
   #         Nu = np.dot(W, X[k,:].T)  # n1 x N
   #         X1 = mlp.phi(Nu)
   #         yn[k+p] = np.dot(V, X1)
   #
   #         #    pass
   #     return yn#

def batch_MLP(method,nodes,X,yr,p,win,re_learn,epoch1,epoch2,learning_rate):
    L = int(yr.shape[0])
    X = nn.input_shape(X)
    yr = nn.output_shape(yr)
    """The dimension of neuron and all matrixes like weights and so on
    depends on number of columns of input vector"""
    X = nn.X_bias(X)
    DIM = nn.input_dim(X[0, :])
    """Creating Neuron object with particular dimension"""

    mlp = nn.MLP(DIM,nodes)
    W = mlp.W
    V = mlp.V
    BP = nn.Backpropagation()

    # CGD u MLP NEFUNGUJE
    CGD = BP.CGD()
    """Creating Method object(BackPropagation) with particular method"""
    yn = np.zeros(L + p)
    VV = np.zeros((L+p,nodes))

    SSE = 0
    if method == 'LM':
        for i in range(epoch1): # LM algoritmus, je potreba prepocitavat Jakobian
            J, e = mlp.Jacobian(W, V, X[:win,:], yr[p:p + win], win) # Spocita Jakobianu a chyby, viz neural_network.py
            W, V = mlp.W_V_LM(W, V, J, learning_rate, e) # nauceni
            SSEE= np.dot(e, e)
            SSE=np.append(SSE,SSEE)

    # Nefunguje - a ani pry nemuze fungovat
    if method == 'CGD':
        J, e = mlp.Jacobian(W, V, X[:win, :], yr[p:p + win], win)
        A = CGD.A(J)
        b = CGD.b(yr[p:p + win], J)
        col_W = mlp.pack_WV(W, V)
        re = CGD.re(b, A, col_W)
        pp = re
        for epoch in range(epoch2):
            J, e = mlp.Jacobian(W, V, X[:win, :], yr[p:p + win], win)
            A = CGD.A(J)
            col_W = mlp.pack_WV(W,V)
            col_W, pp, re = CGD.CGD(A, col_W, re, pp)
            W, V = mlp.unpack_WV(col_W)


    """Sliding window"""
    for k in range(win+p, L ):
        """Re-Learn every x samples, here we relearn every day"""
        if k % (re_learn) == 0 and k >=win+p:  # Retrain
            print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')
            J, e = mlp.Jacobian(W, V, X[k - win - p + 1:k - p + 1, :], yr[k - win+1:k+1], win)
            A = CGD.A(J) # CGD nefunguje
            b = CGD.b(yr[p:p + win], J) # CGD nefunguje
            col_W = mlp.pack_WV(W, V) # CGD nefunguje
            re = CGD.re(b, A, col_W) # CGD nefunguje
            pp = re # CGD nefunguje
            for i in range(epoch2):
                if method== 'LM':
                    J, e = mlp.Jacobian(W, V, X[k - win - p + 1:k - p + 1, :], yr[k - win+1:k+1], win)
                    W, V = mlp.W_V_LM(W, V, J, learning_rate, e)

                print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')

                if method == 'CGD':
                    J, e = mlp.Jacobian(W, V, X[k - win - p + 1:k - p + 1, :], yr[k - win+1:k+1], win)
                    A = CGD.A(J)
                    col_W, pp, re = CGD.CGD(A, col_W, re, pp)
                    W,V = mlp.unpack_WV(col_W)

                SSEE = np.dot(abs(e),abs(e))
                SSE = np.append(SSE, SSEE)
        VV[k,:] = V # ?
        v = np.dot(W, X[k:k + 1, :].T)  # n1 x N
        phi = mlp.phi(v)
        y_temp = mlp.Y(V, phi)
        yn[k+p] = y_temp[-1]

    return yn,SSE

# NGD/GD sample HONU
def XGD_sample_MLP(type,nodes,X,yr,p,win,re_learn,epoch1,epoch2,learning_rate): # NGD = 0.005, GD = NGD/10

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

    e = np.zeros(win)
    SSE = 0
    for epoch in range(epoch1):
        Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X[:win], yr[p:p + win], win) # spocitani
        for i in range(win):
            V = BP.NGD_MLP(e[i], learning_rate, Jv[i, :], V)
            # dv = np.dot(np.dot(np.linalg.inv((np.dot(Jv.T, Jv) + 1. / muv * Lv)), Jv.T), e)
            # V = V + dv

            for nod in range(nodes):
                Jw = Jww[:, nod, :]
                W[nod, :] = BP.NGD_MLP(e[i], learning_rate, Jw[i, :], W[nod, :])

        SSE= np.append(SSE,np.dot(e, e))

    """Sliding window"""
    for k in range(win + p, L):
        """Re-Learn every x samples, here we relearn every day"""
        if k % (re_learn) == 0 and k >= win + p:  # Retrain
            # XX = X[k - win-p+1:k-p+1, :]
            # J = XNU_B.Jacobian(XX)
            X_= X[k - win - p + 1:k - p + 1, :]
            Jww, Jv, e, e_new = mlp.Jw_Jv(W, V, X_, yr[k - win+1:k+1], win)
            for epoch in range(epoch2):
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
                SSE = np.append(SSE, np.dot(e, e))

            print('Finished: ' + str(round((100.0 * ((k * 1.0 - win + 1)) / (L + 1 - win)), 2)) + '%')



        """For prediction we only need one row of samples as they include all reccurent values"""
        Nu = np.dot(W, X[k,:].T)  # n1 x N
        X1 = mlp.phi(Nu)
        yn[k+p] = np.dot(V, X1)

        #    pass
    return yn,SSE


from matplotlib.pyplot import *
"""
Na konci kodu -> spusteni prikladu pouziti
"""

# Priklad 1
def example_1():
    N = 3000
    rand = np.random.rand(N)
    X = np.linspace(0,45,N)
    sum = np.random.rand(X.shape[0])/15

    X = 0.5*np.sin(2*np.pi*X*0.4)+0.3*np.sin(2*np.pi*X*0.8) + sum
    max = np.amax(X)
    X = X/max

    yr = X
    n=10
    X = nn.force_recurrent_gap(X,n,5,0)


    p = 100
    win = 200

    yn,SSE = XGD_sample_MLP('NGD',10,X, yr, p, win, 40, 30, 15, 0.01)


    divide = 1
    subplot(2, 1, 1)
    plot(yr * max, 'k')
    plot(yn * max, 'g')
    err = (yr - yn[:yr.shape[0]]) * max

    plot(err, 'r')
    subplot(2, 1, 2)
    plot(SSE)
    show()
#example_1()

# Priklad 2
def example_2():
    X = np.linspace(0,70,2400)
    sum = np.random.rand(X.shape[0])/5
    X = 0.5*np.sin(2*np.pi*X*0.4)+0.3*np.sin(2*np.pi*X*0.5)# + sum

    p=50
    win=300
    yr = X

    X = nn.force_recurrent_gap(X,15,5,0)

    yn,SSE=batch_MLP('LM',5,X,yr,p,win,10,50,3,0.005)
    #plot(J.T)
    show()
    divide = 10
    subplot(2,1,1)
    title('Predikce')
    plot(yr*divide,'k')
    plot(yn*divide,'g')
    err = (yr-yn[:yr.shape[0]])*divide
    plot(err,'r')

    subplot(2,1,2)
    plot(SSE)
    title('SSE')
    #err = coarse(err,12,24*4,[0,100],3)
    #plot(np.sqrt(err**2),'r')
    show()
#example_2()

# Priklad 3 - vsazena chyba
def example_3():
    dt=0.01
    X = np.linspace(0,30,1000)
    sum = np.random.rand(X.shape[0])

    X = 2*np.sin(2*np.pi*X*2.1)+3*np.sin(2*np.pi*X*0.3) + 1*np.sin(2*np.pi*X*0.77)#+ sum
    #X = X[::2]
    p=50
    win=100
    yr = X
    yr[700:706] = 0
    X = nn.force_recurrent_gap(X,15,5,0)
    print(X.shape)
    #yn = LM_batch('LNU',X,yr,p,win,30,10,10,0.05)
    yn,EA,EAP = CGD_batch('LNU',X,yr,p,win,5,30,3)
    divide = 1
    subplot(3,1,1)
    plot(yr*divide,'k')
    plot(yn*divide,'g')
    err = (yr-yn[:yr.shape[0]])*divide
    plot(err,'r')
    subplot(3,1,2)
    plot(EA)
    subplot(3,1,3)
    plot(EAP)

    #err = coarse(err,12,24*4,[0,100],3)
    #plot(np.sqrt(err**2),'r')

    #X = np.linspace(0,30,8177*2)
    #sum = np.random.rand(X.shape[0])/5
    #X = 2*np.sin(2*np.pi*X*2.1)+3*np.sin(2*np.pi*X*0.3)+ sum + 1*np.sin(2*np.pi*X*0.77)+ sum
    #plot(X,'y')
    show()
#example_3()

# Priklad 4
def example_4():
    dt=0.01
    X = np.linspace(0,30,600)
    sum = np.random.rand(X.shape[0])/3
    X = 2*np.sin(2*np.pi*X*2.1)+3*np.sin(2*np.pi*X*0.3) + 1*np.sin(2*np.pi*X*0.57)*X+ sum
    #X = X[::2]
    p=50
    win=100
    yr = X
    X = nn.force_recurrent_gap(X,10,3,0)
    print(X.shape)
    yn,SSE = NGD_sample_HONU('LNU',X,yr,p,win,20,30,50,0.05)
    divide = 1
    subplot(2,1,1)
    plot(yr*divide,'k')
    plot(yn*divide,'g')
    err = (yr-yn[:yr.shape[0]])*divide
    #err = coarse(err,12,24*4,[0,100],3)
    #plot(np.sqrt(err**2),'r')
    plot(err,'r')
    subplot(2,1,2)
    plot(SSE)

    #X = np.linspace(0,30,8177*2)
    #sum = np.random.rand(X.shape[0])/5
    #X = 2*np.sin(2*np.pi*X*2.1)+3*np.sin(2*np.pi*X*0.3)+ sum + 1*np.sin(2*np.pi*X*0.77)+ sum
    #plot(X,'y')
    show()

def example_5():
    X = np.linspace(0,30,1000)
    sum = np.random.rand(X.shape[0])/3
    X = 2*np.sin(2*np.pi*X*2.1)+3*np.sin(2*np.pi*X*0.3) + 1*np.sin(2*np.pi*X*0.77)+ sum
    p=30
    win=200
    yr = X
    X = nn.force_recurrent_gap(X,15,5,0)
    #print(X.shape)
    yn = LM_batch('LNU',X,yr,p,win,30,10,10,0.05)
    #yn,EA,EAP = CGD_batch('LNU',X,yr,p,win,5,30,3)
    #yn = LM_batch('LNU',X,yr,p,win,30,10,10,0.05)

    divide = 1
    subplot(1,1,1)
    plot(yr*divide,'k')
    plot(yn*divide,'g')
    err = (yr-yn[:yr.shape[0]])*divide
    plot(err,'r')
    show()



"""
spusteni algoritmu
"""
example_3() # vsazena chyba + entropie - pekny priklad
example_4()
example_5()
example_1()
example_2()
