import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import cv2

class MLP(object): # Trida pro MLP site

    def __init__(self,inputLayerSize,nodes): # inputLayerSize - DIM (viz funkce DIM), nodes - pocet neuronu ve skryte vrstve
        '''
        Obejkt MLP site
        :param inputLayerSize: DIM
        :param nodes: pocet neuronu ve skryte vrstve
        '''
        self.inputLayerSize = inputLayerSize
        self.nodes = nodes

        self.W = np.random.randn(self.nodes, self.inputLayerSize) # Vytvoreni matice vah s nalezejicimi rozmery
        self.V = np.random.randn(self.nodes+1) # Vytvoreni vystupni matice vah s nalezejicimi rozmery
        self.W = np.random.randn(self.nodes, self.inputLayerSize )/ self.nodes # normalizace vah
        self.V = np.random.rand(self.nodes) /self.nodes

        # neccesary
        self.nWv = nodes * (self.inputLayerSize) + nodes # Specialni vektor vah (W i V) pro LM ucici algoritmus
        self.col_W = np.random.rand(self.nWv) # Specialni vektor vah (W i V) pro LM ucici algoritmus
        self.L = np.eye(self.nWv) # Jednotkova matice s rozmerem nWv

    def X_bias(self,X_): # funkce pro pridani Biasu
        '''
        Pridani vychyleni
        :param X_: Vstupni matice nebo vektor(musi projit transformaci)
        :return: Vstupni matice nebo vektor s biasem
        '''
        X_ = np.insert(X_, 0, 1, axis=1)
        return X_
    def v(self,W,X): # v = nu , maticove vynasobeni vstupnich vah s vstupnim vektorem
        '''
        Agregace vstupu
        :param W: Vstupni vahy
        :param X: vstupni matice is  Biasem
        :return:
        '''
        v = np.dot(W, X.T)
        return v
    def phi(self,v): # Aktivacni funkce FIRST OUTPUT 1) X_bias -> v -> phi(v) , phi = dy/dv
        '''
        Aktivacni funkce
        :param v: vystup agregace funkce v (nu)
        :return: vystup aktivacni funkce
        '''
        phi_ = 2.0 / (1.0 + np.exp(-v)) - 1.0
        return phi_
    def Y(self,V,phi): # vystup z neuronu
        '''
        vystup ze site - somaticka operace(linearni funkce)
        :param V: vystupni vahy
        :param phi: vystup z aktivacni funkce
        :return: vystup z neuronu
        '''
        y_n=np.dot(V,phi)
        return y_n
    def dphidv(self,v): # druha derivace dphi/dv
        '''
        Derivace aktivacni funkce phi -> dphi/dv
        :param v: vystup agregacni funkce (nu)
        :return: derivace aktivacni funkce
        '''
        dphidv_=2*np.exp(-v)/(1.+np.exp(-v))**2
        return dphidv_
    def test(self,v):
        dphidv_ = 2 * np.exp(-v) / (1. + np.exp(-v)) ** 2
        return dphidv_
    def dydw(self,v,dphidv,X): # derivace vystupu a vstupnich vah dy/dw
        '''
        # derivace vystupu a vstupnich vah dy/dw
        :param v: vystup agregacni funkce
        :param dphidv: vystup derivace aktivacni funkce phi -> dphi/dv
        :param X: vstupni matice
        :return: dy/dw
        '''
        v = input_shape(v * dphidv)
        X = input_shape(X).T
        dydw = v * X
        return dydw

    def Jv(self,v): # Jv = phi, Jakobian
        '''
        Jakobian - viz funkce phi
        :param v:
        :return:
        '''
        Jv = 2.0 / (1. + np.exp(-v)) - 1.0
        return Jv

    def Jw(self,v,dphidv,X): # Jw = dydw,Jakobian
        '''
        Jakobian - viz funkce dydw
        :param v:
        :param dphidv:
        :param X:
        :return:
        '''
        v = input_shape(v*dphidv)
        X = input_shape(X).T
        dydw = v*X
        #dydw = np.dot(np.dot(  V,dphidv ),X)
        return dydw

    def Jacobian(self,W, V, X, yr, win): # specialni jakobian pro LM algoritmus, vyuziva vektoru vah nWv
        '''
        Specialni jakobian - do jednoho vektoru se zabaly vystup z funkci Jv a Jw
        :param W:
        :param V:
        :param X:
        :param yr:
        :param win:
        :return:
        '''
        J = np.zeros((win, self.nWv)) # inicializace
        Nu = np.dot(W, X.T)  # n1 x N
        ####################### N = WINDOW
        X1 = self.phi(Nu)  # 1st layer output for all k
        yn = np.dot(V, X1)
        e = yr - yn
        pom = 0
        dPhidNuX = np.zeros((win, self.inputLayerSize))

        for i in range(self.nodes):
            for k in range(win):
                dPhidNuX[k, :] = self.dphidv(Nu[i, k]) * X[k, :]
            dPhidNuX = V[i] * dPhidNuX
            for j in range(self.inputLayerSize):
                J[:, pom] = dPhidNuX[:, j]
                pom += 1

        for i in range(self.nodes):
            J[:, self.nodes * self.inputLayerSize + i] = X1[i, :]

        Nu = np.dot(W, X.T)  # n1 x N
        ####################### N = WINDOW
        X1 = self.phi(Nu)  # 1st layer output for all k
        yn_new = np.dot(V, X1)
        return J, e#,yn,yn_new
    def Jw_Jv(self,W, V, X, yr, win):
        '''
        Vypocet jakobianu Jw a Jv
        :param W: Vstupni vahy
        :param V: Vystupni vahy
        :param X: vstupni matice
        :param yr: namerene hodnoty
        :param win: velikost uciciho okna
        :return: Jw, Jv, e - chyba mezi vypoctenou a namerenou hodnotou
        '''
        Jv = np.zeros((win, self.nodes))
        Jw = np.zeros((win, self.nodes,self.inputLayerSize))

        Nu = np.dot(W, X.T)  # n1 x N
        ####################### N = WINDOW
        X1 = self.phi(Nu)  # 1st layer output for all k
        yn = np.dot(V, X1)
        e = yr - yn
        pom = 0
        dPhidNuX = np.zeros((win, self.inputLayerSize))

        for i in range(self.nodes):
            for k in range(win):
                dPhidNuX[k, :] = self.dphidv(Nu[i, k]) * X[k, :]
            dPhidNuX = V[i] * dPhidNuX
            for j in range(self.inputLayerSize):
                Jw[:,i,j] = dPhidNuX[:, j]
                pom += 1

        for i in range(self.nodes):
            Jv[:, i] = X1[i, :]

        Nu = np.dot(W, X.T)  # n1 x N
        ####################### N = WINDOW
        X1 = self.phi(Nu)  # 1st layer output for all k
        yn_new = np.dot(V, X1)
        e_new = yr - yn_new
        return Jw,Jv,e,e_new



    def W_V_LM(self,W, V, Jj, mu, e):
        dWv = np.dot(np.dot(np.linalg.inv((np.dot(Jj.T, Jj) + 1. / mu * self.L)), Jj.T), e)

        pom = 0
        for i in range(self.nodes):
            for j in range(self.inputLayerSize):
                W[i, j] += dWv[pom]
                pom += 1
        for i in range(self.nodes):
            V[i] += dWv[self.nodes * self.inputLayerSize + i]
        return W, V


    def unpack_WV(self,col_W): # Nepouziva se
        W_ = self.W
        V_ = self.V
        pom = 0
        for i in range(self.nodes):
            for j in range(self.inputLayerSize):
                W_[i, j] = col_W[pom]
                pom += 1
        for i in range(self.nodes):
            V_[i] = col_W[self.nodes * self.inputLayerSize + i]
        return W_, V_
    def pack_WV(self,W,V): # Nepouziva se
        pom = 0
        col_W_ = np.zeros(self.nWv)
        for i in range(self.nodes):
            for j in range(self.inputLayerSize):
                col_W_[pom] = W[i, j]
                pom += 1
        for i in range(self.nodes):
            col_W_[self.nodes * self.inputLayerSize + i]=V[i]
        return col_W_







#Class of different types of HONU neural units. Divison LNU/QNU/CNU and SampleBySample/Batch
#This class is made in such a way that switching LNU/QNU/CNU can be done by simply switching of all names LNU/QNU/CNU in a script

# V teto tride komentar pouze k QNU_Batch a k QNU sample, nebot se vse dale opakuje
class HONU(object): # HONU trida
    class QNU_Batch(object): # Podtrida QNU - argument je DIM viz funkce input_DIM.
        # PRO DAVKOVE UCENI (+Jakobian, jinak podobne jako Sample)
        '''
        Object QNU neuronu pro davkove uceni
        '''
        def __init__(self, inputLayerSize):
            '''
            Inicializace objektu
            :param inputLayerSize: velikost neuronu (vychazi z DIM - tak jak jej znacim v main.py- velikost vstupniho vektoru bez vychyleni/biasu)
            '''
            self.inputLayerSize = inputLayerSize  # excluding x0, that means n, so therefore real size is (n+1) -> including x0 (bias)

            # Weights (parameters) - je mozne si vybrat nekterou z vah pri inicializaci objektu, nejcasteji pouzivane jsou samozrejme nahodne vahy W_QNU_init
            self.W_QNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1) #Vahy s nalezejicimi rozmery
            self.W_QNU_zeros = np.zeros((self.inputLayerSize + 1, self.inputLayerSize + 1)) #Nahodne vahy s nalezejicimi rozmery
            self.W = np.ones((self.inputLayerSize + 1, self.inputLayerSize + 1))  #Jednotkove Vahy s nalezejicimi rozmery ()

            self.Y_QNU = None
            self.length = (self.inputLayerSize + 1) * (
                self.inputLayerSize + 2) / 2  # Vypocet prvku matice v hornim trojuhelniku - rozmer dlouheho plocheho vektoru

        def Jacobian(self,X_): # Vypocet jakobianu - u HONY vhodne vypocet Jakobian najednou, nebot se nemenni (Zavisi pouze na vstupech do neuronu)
            '''
            Jacobian
            :param X_: Vstupni matice - neni treba nic jineho
            :return: J -> Jakobian
            '''
            # ADD BIAS
            X_=np.insert(X_,0,1,axis=1)

            # INICIALIZATION

            J = np.zeros((int(X_.shape[0]), self.length)) # rozmery dle vstupniho vektoru a dlouheho plocheho vektoru
            col_X = np.zeros(self.length) # vstupni vektor - dlouhy plochy vektor


            # Algorithm for J
            counter = 0
            counter_jac = 0
            for X in X_:
                for i in range(self.inputLayerSize + 1):
                    for j in range(i, self.inputLayerSize + 1):
                        # Y_QNU[counter] = W[i,j]*X[i]*X[j] # SECOND OPTION
                        col_X[counter] = X[i] * X[j]
                        counter += 1

                counter = 0
                J[counter_jac] = col_X
                counter_jac += 1
            return J

        def col_W_from_W(self,W): # agregacni funkce - pouze vahy - z matice vah vytvorim dlouhy plochy vektor vah col_W
            '''
            Prevod z Matice vah na dlouhy plochy vektor (viz diplomova prace). 2D->1D
            :param W:
            :return: col_W
            '''
            counter = 0
            col_W = np.zeros(self.length)
            for i in range(self.inputLayerSize + 1):
                for j in range(i, self.inputLayerSize + 1):
                    col_W[counter] = W[i, j]
                    counter += 1

            return col_W

        def Y(self, J, col_W): # vystup z neuronu
            """
            maticovy soucin -> vystup neuronu
            :param J: Jakobian
            :param col_W: dlouhy plochy vektor vah
            :return:  vystup z neuronu
            """
            try:
                Y_QNU = np.dot(J, col_W)
            except IndexError:
                print('Wrong matrix dimensions')

            return Y_QNU

    class QNU_Sample(object): # dle QNU_Batch
        '''
        Objekt QNU neuronu pro krokove uceni
        '''
        def __init__(self, inputLayerSize):
            """
            Inicializace Objektu
            :param inputLayerSize: velikost neuronu - DIM - velikost vstupniho vektoru bez vychyleni
            """
            self.inputLayerSize = inputLayerSize

            # Weights (parameters)
            self.W_QNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1) # Matice VAH
            self.W_QNU_zeros = np.zeros((self.inputLayerSize + 1, self.inputLayerSize + 1)) # Matice VAH
            self.W = np.ones((self.inputLayerSize + 1, self.inputLayerSize + 1))  # Matice VAH
            self.Y_QNU = None
            self.length = (self.inputLayerSize + 1) * (
            self.inputLayerSize + 2) / 2  # Vypocet prvku matice v hornim trojuhelniku - dlouhy plochy vektor

        def col_X(self,X): # argument neni vstupni matice ale radek vstupni matice - tedy vstupni vektor
            """
            Agregace vstupu -> v podstate jakobian, ale jen s jednim radkem
            :param X: Aktualni radek vstupni matice
            :return: dlouhy plochy vektor vstupu
            """
            X = np.insert(X, 0, 1)

            try:
                # INICIALIZATION
                col_X = np.zeros(self.length)
                #Algorithm
                counter = 0
                for i in range(self.inputLayerSize + 1):
                    for j in range(i, self.inputLayerSize + 1):
                        col_X[counter] = X[i] * X[j]

                        counter += 1
                return col_X

            except IndexError:
                print('Wrong matrix dimensions')

        def col_W_from_W(self,W):
            '''
            Prevod z Matice vah na dlouhy plochy vektor (viz diplomova prace). 2D->1D

            :param W: Vahy
            :return: col_W
            '''
            try:
                # INICIALIZATION
                col_W = np.zeros(self.length)
                # Algorithm - agregacni funkce
                counter = 0
                for i in range(self.inputLayerSize + 1):
                    for j in range(i, self.inputLayerSize + 1):
                        # Y_QNU[counter] = W[i,j]*X[i]*X[j] # SECOND OPTION
                        col_W[counter] = W[i,j]
                        counter += 1
                return col_W
            except IndexError:
                print('Wrong matrix dimensions')

        def Y(self,col_X,col_W):
            Y = np.dot(col_X,col_W)
            return Y

    class LNU_Sample(object):
        def __init__(self, inputLayerSize):
            self.inputLayerSize = inputLayerSize  # including x0, that means n, so therefore real size is (n+1)

            # Weights (parameters)
            self.W_LNU_init = np.random.randn(self.inputLayerSize + 1)
            self.W_LNU_zeros = np.zeros((self.inputLayerSize + 1))

            self.W = np.ones((self.inputLayerSize + 1, self.inputLayerSize + 1))  # Matice VAH
            self.length = (self.inputLayerSize + 1) * (
            self.inputLayerSize + 2) / 2  # Vypocet prvku matice v hornim trojuhelniku

        def col_X(self,X):

            X = np.insert(X, 0, 1)  # .reshape(shapeX)
            try:
                # INICIALIZATION
                col_X = np.zeros(self.inputLayerSize + 1)
                #Algorithm
                for i in range(self.inputLayerSize + 1):
                    col_X[i] = X[i]
                return col_X

            except IndexError:
                print('Wrong matrix dimensions')

        def col_W_from_W(self, W):
            try:
                # INICIALIZATION
                col_W = np.zeros(self.inputLayerSize+1)
                # Algorithm
                counter = 0
                for i in range(self.inputLayerSize + 1):
                    col_W[counter] = W[i]
                    counter += 1
                return col_W
            except IndexError:
                print('Wrong matrix dimensions')

        def Y(self,col_X,col_W):
            Y = np.dot(col_X,col_W)
            return  Y

    class LNU_Batch(object):
        def __init__(self, inputLayerSize):
            self.inputLayerSize = inputLayerSize  # including x0, that means n, so therefore real size is (n+1)

            # Weights (parameters)
            self.W_QNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1)
            self.W_LNU_init = np.random.randn(self.inputLayerSize + 1)
            self.W_LNU_zeros = np.zeros((self.inputLayerSize + 1))

            self.W = np.ones((self.inputLayerSize + 1, self.inputLayerSize + 1))  # Matice VAH
            self.Y_QNU = None
            self.length = (self.inputLayerSize + 1) * (
            self.inputLayerSize + 2) / 2  # Vypocet prvku matice v hornim trojuhelniku

        def Jacobian(self,X_):

            # ADD BIAS
            X_=np.insert(X_,0,1,axis=1)

            # INICIALIZATION

            J = np.zeros((int(X_.shape[0]), self.inputLayerSize+1))
            col_X = np.zeros(self.inputLayerSize+1)

            # Algorithm for J
            counter = 0
            counter_jac = 0

            for X in X_:
                for i in range(self.inputLayerSize + 1):
                    col_X[counter] = X[i]
                    counter += 1
                counter = 0
                J[counter_jac] = col_X
                counter_jac += 1
            #print(J.shape)
            return J

        def col_W_from_W(self,W):
            counter = 0
            col_W = np.zeros(self.inputLayerSize + 1)
            for i in range(self.inputLayerSize + 1):
                    col_W[counter] = W[i]
                    counter += 1
            return col_W

        def Y(self, J, col_W):
            try:
                Y_QNU = np.dot(J, col_W)
            except IndexError:
                print('Wrong matrix dimensions')
            return Y_QNU
    # CNU BATCH FINISHED
    class CNU_Batch(object):
        def __init__(self, inputLayerSize):
            self.inputLayerSize = inputLayerSize  # including x0, that means n, so therefore real size is (n+1)

            # Weights (parameters)
            self.W_CNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1, self.inputLayerSize + 1)
            self.W_QNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1)
            self.W_LNU_init = np.random.randn(self.inputLayerSize + 1)
            self.W_CNU_zeros = np.zeros((self.inputLayerSize + 1,self.inputLayerSize + 1,self.inputLayerSize + 1))

            self.W = np.ones((self.inputLayerSize + 1, self.inputLayerSize + 1))  # Matice VAH
            self.Y_CNU = None
            self.length = (inputLayerSize+1)*(inputLayerSize+2)*(inputLayerSize+3)/6  # Vypocet prvku matice v hornim trojuhelniku

        def Jacobian(self, X_):

            # ADD BIAS
            X_ = np.insert(X_, 0, 1, axis=1)

            # INICIALIZATION

            J = np.zeros((int(X_.shape[0]), self.length))
            col_X = np.zeros(self.length)

            # Algorithm for J
            counter = 0
            counter_jac = 0

            for X in X_:
                for i in range(self.inputLayerSize + 1):
                    for j in range(i, self.inputLayerSize + 1):
                        for k in range(j,self.inputLayerSize + 1):
                            # Y_QNU[counter] = W[i,j]*X[i]*X[j] # SECOND OPTION
                            col_X[counter] = X[i] * X[j] * X[k]
                            counter += 1

                counter = 0
                J[counter_jac] = col_X
                counter_jac += 1
            return J

        def col_W_from_W(self, W):
            counter = 0
            col_W = np.zeros(self.length)
            for i in range(self.inputLayerSize + 1):
                for j in range(i, self.inputLayerSize + 1):
                    for k in range(j, self.inputLayerSize + 1):
                        # Y_QNU[counter] = W[i,j]*X[i]*X[j] # SECOND OPTION
                        col_W[counter] = W[i, j, k]
                        # print(i,j)
                        counter += 1
            # print(col_W.shape)
            # try:
            #    col_W.shape[1]
            # except IndexError:
            #    col_W = col_W.reshape(col_W.shape[0],1)
            '''Problem:
            pokud pouziji data s jednim vstupem, pak col_W s shapem [R,] nefunguje, pokud udelam reshape na [R,1],
            pak operace np.dot(A,col_W) misto cisla vytvori matici a tim padem vznikne spatny format reziduaa(ktery ma bejt cislo).
            RESENI. Data s jednim vstupem reshapnu z [R,] na [R,1].
            Problem. Mam matici 3x3. pokud ji roznasobim s [3,] vznikne mi [,3]
                     Mam matici 3x3. pokud ji roznasobim s [3,1] vznikne mi [3,1]

            '''

            return col_W

        def Y(self, J, col_W):
            try:
                Y_QNU = np.dot(J, col_W)
            except IndexError:
                print('Wrong matrix dimensions')

            return Y_QNU



    class CNU_Sample(object):
        def __init__(self, inputLayerSize):
            self.inputLayerSize = inputLayerSize  # including x0, that means n, so therefore real size is (n+1)

            # Weights (parameters)
            self.W_CNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1,
                                              self.inputLayerSize + 1)
            self.W_QNU_init = np.random.randn(self.inputLayerSize + 1, self.inputLayerSize + 1)
            self.W_LNU_init = np.random.randn(self.inputLayerSize + 1)
            self.W_CNU_zeros = np.zeros((self.inputLayerSize + 1,self.inputLayerSize + 1, self.inputLayerSize + 1))

            self.W = np.ones((self.inputLayerSize + 1, self.inputLayerSize + 1))  # Matice VAH
            self.Y_CNU = None
            self.length = (inputLayerSize + 1) * (inputLayerSize + 2) * (
            inputLayerSize + 3) / 6  # Vypocet prvku matice v hornim trojuhelniku

        def col_X(self, X):

            X = np.insert(X, 0, 1)  # .reshape(shapeX)

            try:
                # gaussuv soucet - pocet prvku v hornim trojuhelniku matice


                # INICIALIZATION
                col_X = np.zeros(self.length)
                # Algorithm
                counter = 0
                for i in range(self.inputLayerSize + 1):
                    for j in range(i, self.inputLayerSize + 1):
                        # Y_QNU[counter] = W[i,j]*X[i]*X[j] # SECOND OPTION
                        col_X[counter] = X[i] * X[j]
                        # print(i,j)
                        counter += 1
                return col_X

            except IndexError:
                print('Wrong matrix dimensions')

        def col_W_from_W(self, W):
            try:
                # INICIALIZATION
                col_W = np.zeros(self.length)
                # Algorithm
                counter = 0
                for i in range(self.inputLayerSize + 1):
                    for j in range(i, self.inputLayerSize + 1):
                        for k in range(j, self.inputLayerSize + 1):
                            # Y_QNU[counter] = W[i,j]*X[i]*X[j] # SECOND OPTION
                            col_W[counter] = W[i, j, k]
                            # print(i,j)
                            counter += 1
                return col_W
            except IndexError:
                print('Wrong matrix dimensions')

        def Y(self, col_X, col_W):
            Y = np.dot(col_X, col_W)
            return Y


'''
Prace s vektorem/matici vstupu
'''
#Trida s kterou lze manipulovat se vstupni matici -> Asi zbytecne komplikovane naprogramovana, vychazi z ni funkce force_recurrent....
# Asi neni treba komentovat
class Dynamic_batch(object):
    def __init__(self,yr,number_of_reccurents,initial):
        self.input = np.ones((int(yr.shape[0]),number_of_reccurents))*initial
    def init_matrix(self):
        return self.input
    def input_history(self,input,yr):
        input[:,0] = yr[:]
        for i in range(1,int(input.shape[1])):
            #reverse the numbers
            input[i:,i] = yr[:-i]
        return input


def recurrent(X,win,n): #funkce ktera udela ze vstupniho vektoru vstupni matici se zpozdenymi hodnotami
    # pozor - zpozdene hodnoty musi existovat !
    # argument win - rozmer vystupni matice win X n
    '''
    #funkce ktera udela ze vstupniho vektoru vstupni matici se zpozdenymi hodnotami
    :param X: Vstupni vektor o rozmeru Z
    :param win: velikost vysledne vstupni matice -> win < Z-n
    :param n: pocet zpozdenych hodnot
    :return: vstupni matice
    '''
    new_X = np.zeros((win,n))
    for i in range(n):
        new_X[:,i] = X[-win-i:-i+X.shape[0]]
    return new_X
# Array adjustment without need of making object
def force_recurrent(X,n1,init):
    # zpozdene hodnoty nemusi existovat
    # vysledna vystiupni matice ma stejny rozmer jako vstupni vektor(argument)
    '''
    #funkce ktera udela ze vstupniho vektoru vstupni matici se zpozdenymi hodnotami
    :param X: vstupni vektor
    :param n1:  pocet zpozdenych hodnot
    :param init: hodnota neexistujicich zpozdenych hodnot
    :return: vystupni matice
    '''
    DYNAMIC_data = Dynamic_batch(X,n1,init)
    init = DYNAMIC_data.init_matrix()
    X = DYNAMIC_data.input_history(init,X)
    return X
def force_recurrent_gap(X,n1,gap,init): # viz Force_recurrent
    '''
    :param X: vstupni vektor
    :param n1: pocet zpozdenych hodnot
    :param gap: mezera mezi zpozdenymi hodnotami
    :param init: hodnota neexistujicich zpozdenych hodnot
    :return:
    '''
    DYNAMIC_data = Dynamic_batch(X, n1, init)
    init = DYNAMIC_data.init_matrix()
    for i in range(n1):
        init[i*gap:, i] = X[:-i*gap+ X.shape[0]]
    return init

def recurrent_gap(X,win,n,gap): # viz reccurent
    '''
    :param X: Vstupni vektor o velikosti Z
    :param win: rozmer vystupni matice -> pozor win< Z-n*gap
    :param n: pocet zpozdenych hodnot
    :param gap: mezera mezi zpozdenymi hodnotami
    :return:
    '''
    new_X = np.zeros((win, n))
    for i in range(n):
        new_X[:, i] = X[-win - i*gap:-i*gap + X.shape[0]]
    return new_X

"""
Examples
num=np.arange(1000)
x = nn.force_recurrent_gap(num,4,3,0)
y = nn.force_recurrent(num,4,0) 999 998 997 996
z = nn.recurrent_gap(num,20,4,3) 999,996,993,900 last row
"""

'''
Ucici algoritmy(Zejmena pro HONU)
'''
#Class of Backpropagation Methods
class Backpropagation(object):
    def __init__(self):
        pass

    '''Sample methods NGD, GD'''
    # Normalized Gradient Descent
    def NGD(self,y_target,y_neuron,learning_rate,col_X,col_W): # argumenty viz nahore
        '''
        # Normalized Gradient Descent for HONU
        :param y_target: vektor namerenych hodnot
        :param y_neuron: vektor vypoctenych hodnot
        :param learning_rate: rychlost uceni
        :param col_X: vstupni vektor - dlouhy plochy vektor
        :param col_W: vektor vah - dlouhy plochy vektor
        :return:
        '''
        mu = learning_rate
        error = y_target-y_neuron
        uW = mu / (1 + np.dot(col_X,col_X)) * error * col_X #(1 + sum(col_X ** 2)) * error * col_X IVO str.6, 52/53, An approach to Stable Gradient Descent Adaptation of ....
        col_W += uW
        return col_W

    def NGD_MLP(self,error,learning_rate,col_X,col_W):
        '''
        # Normalized Gradient Descent for MLP
        :param error: chyba mezi yr a yn
        :param learning_rate: rychlost uceni
        :param col_X: vstupni vektor - dlouhy plochy vektor
        :param col_W: vektor vah - dlouhy plochy vektor
        '''
        mu = learning_rate
        uW = mu / (1 + np.dot(col_X,col_X)) * error * col_X #(1 + sum(col_X ** 2)) * error * col_X IVO str.6, 52/53, An approach to Stable Gradient Descent Adaptation of ....
        col_W += uW

        return col_W
    #Gradient descent
    def GD_MLP(self, error, learning_rate, col_X, col_W):
        '''
        # Gradient Descent for MLP
        :param error: chyba mezi yr a yn
        :param learning_rate: rychlost uceni
        :param col_X: vstupni vektor - dlouhy plochy vektor
        :param col_W: vektor vah - dlouhy plochy vektor
        :return:
        '''
        mu = learning_rate
        uW = mu * error * col_X
        col_W += uW
        return col_W

    def GD(self,y_target,y_neuron,learning_rate,col_X,col_W):
        '''
        # Gradient Descent for HONU
        :param y_target: vektor namerenych hodnot
        :param y_neuron: vektor vypoctenych hodnot
        :param learning_rate: rychlost uceni
        :param col_X: vstupni vektor - dlouhy plochy vektor
        :param col_W: vektor vah - dlouhy plochy vektor
        :return:
        '''
        mu = learning_rate
        error = y_target-y_neuron
        uW = mu * error * col_X
        col_W += uW
        return col_W


    '''Batch methods CGD, LM '''
    class CGD(object):
        def __init__(self):
            pass
        # Pocatecni promenna

        def b(self,y_target,J):
            '''
            # Pocatecni promenna
            Viz diplomova prace
            :param y_target:
            :param J:
            :return:
            '''
            b = np.dot(J.T , y_target)
            return b

        # Pocatecni promenna - symetricka ctvercova matice
        def A(self,J):
            '''
            # Pocatecni promenna - symetricka ctvercova matice
            Viz diplomova prace
            :param J:
            :return:
            '''
            A=np.dot(J.T,J)
            return A

        # Pocatecni promenna
        def re(self,b,A,col_W):
            '''
            # Pocatecni promenna
            :param b:
            :param A:
            :param col_W:
            :return:
            '''
            re = b - np.dot(A , col_W)
            return re
        #POZOR
        # dalsi pocatecni promena je p=re

        # Iterace -> viz Diplomova prace
        def CGD(self,A,col_W,re,p):
            '''
            Iterace -> Viz diplomova prace
            :param A: vstup
            :param col_W: vstup
            :param re: vstup
            :param p: vstup
            :return: col_W,p,re -> prijde v dalsim kole jako vstup
            '''
            #p == re pro p.p. 0
            residuum = np.dot(re.T,re)
            #Problemy pokud je vystup neuronu Y = [R,1], musi byt Y = [R,]. maticovej nesmysl pri nasobeni nebo np.dot
            Ap = np.dot(A,p)
            alfa = residuum/np.dot(p.T,Ap)
            re_NEW=re - alfa*Ap
            residuum_NEW = np.dot(re_NEW.T,re_NEW)
            #print(residuum_NEW.shape,residuum.shape)
            if residuum_NEW!=0.0 or residuum !=0: # nebezpeci deleni nulou
                beta = residuum_NEW/residuum
                col_W = col_W + alfa*p
                p_NEW = re_NEW+beta*p

            #print('Error: Propably too low beta which is division between residiuums (k+1)/k')
                return col_W,p_NEW,re_NEW
            else:
                print('enough training')
                return col_W,p,re

    def LM(self,y_target,y_neuron,learning_rate,col_W,J):
        '''
        Levenberg-Marquardt
        :param y_target: namerezene hodnoty
        :param y_neuron: vystup z neuronu
        :param learning_rate: rychlost uceni
        :param col_W: dlouhy plochy vektor bah
        :param J: Jakobian
        :return: col_W,dW
        '''
        try: # odchyceni problemu u numpy, kdy vektory maji rozmer bud R[L,] nebo R[L,1]
            dimension = int(col_W.shape[1])
        except IndexError:
            dimension = int(col_W.shape[0])

        I=np.eye(dimension) # jednotkova matice
        mu = learning_rate
        error = y_target - y_neuron
        dW = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + 1. / mu * I), J.T), error)
        col_W = col_W + dW
        return col_W,dW


"""Functions for easier use of Neural Network module"""
"""
INPUTS AND OUTPUTS FORM:
Columns x1,x2,x3....
Rows x1(k),x1(k+1).... -data in time
"""

# Put X through this function. return DIM, which will calculate dimension for layersize of HONU
# example QNU = HONU().QNU_Batch(DIM)
# In Dynamic HONU we must count with reccurent inputs so total Dimension is DIM+Y_recurrent+U_recurrent - 1 (-1 sometimes)

def input_dim(X): # Pro HONU, Funkce slouzi k ziskani rozmeru vstupni matice ci vektoru X
    try:
        DIM = int(X.shape[1])
    except:
        DIM = int(X.shape[0])
    return DIM # DIM je pak argument pro vytvoreni neuronu HONU

# Use if you have only 1D input
def input_shape(X): # Vzdy pouzit na vstupni(V tomto pripade mysleno na vektor ktery nekam vstupuje) vektor
    """"Vecny Problem, kdy moje prace nekdy bere vektory ve tvaru [R,] a nekdy [R,1]"""
    # TOTO JE RESENI -> Vstup do neuronu prohnat timto
    try:
        X.shape[1]
    except IndexError:
        #print('Input reshaped from shape [R,] to [R,1]')
        X = X.reshape(X.shape[0],1)
    return X

# Use (always) for Y target
def output_shape(Y): # Vzdy pouzit na vektor ktery od nekad vystupuje(napr yn,yr)
    try:
        Y.shape[1]
        Y = Y.reshape(-Y.shape[0])
    except IndexError:
        pass
    return Y

#Pri praci s MLP - prida se sloupec Biasu
def X_bias(X_):
    X_ = np.insert(X_, 0, 1, axis=1)
    return X_


# Coarse
def coarse_grain(yr,radius): # metoda Coarse Graining
    y_cg = np.zeros(radius + yr.shape[0] + radius) # vytvorim vektor, kterej je o 2*radius delsi, nebot se na zacatku i na konci musi vyhlazovat z dat, ktera neexistuji, napr y[0-radius]
    y_cg[radius:radius + yr.shape[0]]=yr #[0 0 0 yr yr yr yr .... yr 0 0 0] ->radius=3
    for i in range(yr.shape[0]):
        y_cg[i+radius]=np.sum(y_cg[i:i+2*radius+1])/(2.*radius+1.)
    return y_cg[radius:-radius]

def stack_input(X,U): # spojovani vstupni matice s dalsimi vstupnimi maticemi
    U = input_shape(U)
    X =np.hstack((X,U))
    return X

## def fcnEA - Learning entropy Author - IVO BUKOVSKY
def fcnEA(Wm, alphas, OLEs):  # Wm ... recent window of weights including the very last weight updates
    OLEs = OLEs.reshape(-1)
    nw = Wm.shape[1]
    nalpha = len(alphas)
    ea = np.zeros(len(OLEs))
    i = 0
    for ole in range(np.max(OLEs) + 1):
        if ole == OLEs[i]:  # assures the corresponding difference of Wm
            absdw = np.abs(Wm[-1, :])  # very last updated weights
            meanabsdw = np.mean(abs(Wm[0:Wm.shape[0] - 1, :]), 0)
            Nalpha = 0
            for alpha in alphas:
                Nalpha += np.sum(absdw > alpha * meanabsdw)
            ea[i] = float(Nalpha) / (nw * nalpha)
            i += 1
        Wm = Wm[1:, :] - Wm[0:(np.shape(Wm)[0] - 1), :]  # difference Wm
    return (ea)


def Learning_entropy(col_W,OLE,L,p,memory): # Vytvoreni potrebnych matic - neopotrebovani kodu
    OLEs = np.array([1, OLE])
    Wm = np.zeros((memory, col_W.shape[0]))
    EA = np.zeros((L + p, OLEs.shape[0]))
    EAP = np.zeros((L + p, OLEs.shape[0]))
    return Wm,EA,EAP,OLEs

def Wm_operation(Wm,col_W): # zarazeni col_W do pameti vah Learning Entropy
    Wm[0:-1, :] = Wm[1:, :]
    Wm[-1, :] = col_W
    return Wm

def arithmetic_error(e,n): # dle diplomove prace
    '''
    :param e: vektor chyb
    :param n: pocet chyb ze kteryh se udela prumerna chyba  k z (k-n,k)
    :return: vektor prumernych chyb
    '''
    L = e.shape[0]
    ari_err =np.zeros(L)
    for k in range(n,L):
        ari_err[k]= np.sum(((e[k-n+1:k+1])**2)**0.5)/n
    return ari_err

def ratio_yr_to_ari_er(value,ari_er):
    error_percentage = ari_er/value
    return error_percentage


#### SELF ORGANIZING MAPS

class SOM(object):
    def __init__(self,n,m,I):
        self.n=n
        self.m=m
        self.I=I # a-pocet -(treba pocet barev), b-struktura (RGB-3)
        self.weights = np.random.rand(n,m,I.shape[1])
        #I = np.array([[1,1,1],[0,1,0],[1,0,0],[0,0,0]])#,[0,1,1]])

        self.winner = np.zeros((n, m)) # saves the Carthese coordinates of actual winner which will be saved in winner_z
        self.winner_z = np.zeros((I.shape[0], 2)) # saves the winners of all inputs in one epochs

    def distance(self,inp,w):
        d= np.sqrt(np.sum((inp-w)**2))
        return d
    def update_w(self,i,ww,g,L):
        ww = ww + g*L*(i-ww)
        return ww
    def n_dist(self,inp,wei):
        dis = np.sqrt(np.sum((inp-wei)**2))
        return dis
    def neighbourhood_size(self,G0,t,lam):
        G=G0**(np.exp(-t/lam))
        return G
    def neighbourhood_size_mh(self,G0,t,lam):
        G=G0*(np.exp(-t/lam))
        return G
    def Le(self,L0,t,lam):
        L=L0*(np.exp(-t/lam))
        return L
    def gauss(self,V1,V2,G): # influence rate
        dis = np.sum((V1 - V2) ** 2)
        gau=np.exp(-dis/(2*(G**2)))
        return gau
    def mexican_hat(self,V1,V2,G): # influence rate
        dis = np.sum((V1 - V2) ** 2)
        mh=(1-np.sqrt(dis)/(G**2))*np.exp(-dis/(2*(G**2)))
        return mh

    def som_it(self,epochs, n, m, G0, L0, tau1, tau2,weights):
        I = self.I
        for ep in range(epochs):
            G = self.neighbourhood_size(G0, ep, tau1)
            L = self.Le(L0, ep, tau2)

            for z in range(I.shape[0]):
                for k in range(n):
                    for l in range(m):
                        self.winner[k, l] = self.distance(I[z], weights[k, l, :])

                win = np.where(self.winner == np.amin(self.winner))
                win = np.asarray(win) * 1.0
                self.winner_z[z, :] = win[:, 0]
            for z in range(I.shape[0]):
                for l in range(m):
                    for k in range(n):
                        g = self.gauss(self.winner_z[z, :], np.array([k, l]), G)
                        weights[k, l, :] = self.update_w(I[z], weights[k, l, :], L, g)
            print(100. * ep / epochs)
        return weights

    def som_one(self,ep, n, m, G0, L0, tau1, tau2,weights,type='mexican'):
        I = self.I
        G = self.neighbourhood_size(G0, ep, tau1)
        L = self.Le(L0, ep, tau2)

        for z in range(I.shape[0]):
            for k in range(n):
                for l in range(m):
                    self.winner[k, l] = self.distance(I[z], weights[k, l, :])

            win = np.where(self.winner == np.amin(self.winner))
            win = np.asarray(win) * 1.0
            self.winner_z[z, :] = win[:, 0]
        for z in range(I.shape[0]):
            for l in range(m):
                for k in range(n):
                    if type =='gauss':
                        g = self.gauss(self.winner_z[z, :], np.array([k, l]), G)
                    if type=='mexican':
                        g = self.mexican_hat(self.winner_z[z, :], np.array([k, l]), G)

                    weights[k, l, :] = self.update_w(I[z], weights[k, l, :], L, g)

        return weights

    def u_matrix(self,weights):
        n = weights.shape[0]
        m = weights.shape[1]
        dM = np.zeros((n + n - 1, m + m - 1))

        def distance(V1, V2):
            d = np.sqrt(np.sum((V1 - V2) ** 2))
            return d

        conn = 0
        ii = 0

        for i in range(n + n - 1):
            jj = 0
            con = 0
            for j in range(m + m - 1):
                if (i + 1) % 2 != 0 and (j + 1) % 2 != 0:
                    av = np.zeros(4)
                    constant = 4.
                    if ii + 1 < n:
                        av[0] = distance(weights[ii, jj], weights[ii + 1, jj])
                    else:
                        constant -= 1
                    if jj + 1 < m:
                        av[1] = distance(weights[ii, jj], weights[ii, jj + 1])
                    else:
                        constant -= 1
                    if ii - 1 >= 0:
                        av[2] = distance(weights[ii, jj], weights[ii - 1, jj])
                    else:
                        constant -= 1
                    if jj - 1 >= 0:
                        av[3] = distance(weights[ii, jj], weights[ii, jj - 1])
                    else:
                        constant -= 1
                    dM[i, j] = np.sum(av) / constant
                if (i + 1) % 2 != 0 and (j + 1) % 2 == 0:
                    dM[i, j] = distance(weights[ii, jj], weights[ii, jj + 1])
                if (i + 1) % 2 == 0 and (j + 1) % 2 != 0:
                    dM[i, j] = distance(weights[ii, jj], weights[ii + 1, jj])
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    dM[i, j] = distance(weights[ii, jj], weights[ii + 1, jj + 1])
                con += 1
                if con == 2:
                    con = 0
                    jj += 1
            conn += 1
            if conn == 2:
                conn = 0
                ii += 1
        return dM

    def plot_net(self,weights):
        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
        X = weights[:, :, 0]
        Y = weights[:, :, 1]
        Z = weights[:, :, 2]
        # ax.surf(X,Y,Z,'FaceAlpha',0.5,'EdgeColor','none')
        ax.plot_wireframe(X, Y, Z, color=[0, 0, 0], alpha=1, linewidth=0.2)
        show()
        return None

    def plot_scatter(self,weights):
        from mpl_toolkits.mplot3d import Axes3D
        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
        X = weights[:, :, 0]
        Y = weights[:, :, 1]
        Z = weights[:, :, 2]
        ax.scatter(X, Y, Z, c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        show()

    def plot_2dnet(self,weights):
        for i in range(weights.shape[0]):
            X = weights[i, :, 0]
            Y = weights[i, :, 1]
            plot(X, Y, 'k')
        for j in range(weights.shape[1]):
            X = weights[:, j, 0]
            Y = weights[:, j, 1]
            plot(X, Y, 'k')
        show()

    def plot_2dnet_data(self, weights,data,type='k',type_data='ro'):
        for i in range(weights.shape[0]):
            X = weights[i, :, 0]
            Y = weights[i, :, 1]
            plot(X, Y, type)
        for j in range(weights.shape[1]):
            X = weights[:, j, 0]
            Y = weights[:, j, 1]
            plot(X, Y, type)
        plot(data[:,0],data[:,1],type_data)
        show()

def translate_image(img):
    I = np.array([0, 0])
    img = cv2.flip(img, 0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.sum(img[i, j, :]) < 2.8:
                I = np.vstack((I, np.array([j, i])))
    try:
        I = I[1:, :]
    except:
        pass
    return I


# HOW TO DO 
#weights = som_it(300,n,m,(n+m)/2,0.5,50,250,I,weights)
#for ep in range(300):
#    weights=som_it(ep,n,m,(n+m)/2,0.5,100,100,I,weights)
#imshow(weights)
#show()
