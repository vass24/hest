import numpy as np
import sys
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt 
import libfmp.c2
import librosa
np.set_printoptions(threshold=sys.maxsize) # PER STAMPARE OGNI VETTORE INTERAMENTE

#IMPOSTAZIONE ACCORDI A QUATTRO VOCI (SE 1 RICONOSCE LE ESTENSIONI ALTRIMENTI NIENTE)
quadricordo = 1

#BPM SET
bpm = 120

#SAMPLERATE SET
s_r = 12000

#CHORD SETTINGS
n_accordi = 8
n_t = (n_accordi * 2) + 1 #prendo il doppio degli excerpt rispetto agli accordi da rilevare, così da avere più precisione e rilevare anche le mezze misure (da provare anche con accordi * 4), il +1 è per i timestamp (ci deve essere anche quello di fine)
spb = 60 / bpm #durata in secondi di ogni beat
gap = spb * 2 #le finestre sono lunghe 2 beat

#"LA" CENTRALE
f_ref = 440

#FFT SIZE
N = 4096 #NUMPY
L_N = N*2 #LIBROSA

#fft (stft)-> MFCC = suddivisione lineare delle frequenze, se dopo si applica una DCT si ottiene il MEL spectrogram, cioè logaritmico

def f_pitch(p): # prende in input una nota midi e restituisce la frequenza corrispondente es. p = 69 // return 440
    return (2 ** ((p - 69) / 12)) * f_ref 

def pool_pitch(p, Fs, N): # input: p intero passato dalla funzione sotto che rappresenta la nota midi tra 0 e 127, SR, N CAMPIONI // output: array con elenco di indici indicanti le frequenze appartenenti a quella nota p
    lower = f_pitch(p - 0.5) # trova la frequenza minore che riconosce la nota in questione
    upper = f_pitch(p + 0.5) # trova la frequenza maggiore che riconosce la nota in questione
    k = np.arange(N // 2 + 1) # crea un array di 2049 elementi poiché N=4096 -> [0,1,2,3,4,5,6,7,8....,2047,2048], cioè l'fft size
    k_freq = k * Fs / N  # F_coef(k, Fs, N) // k_freq rappresenta tutte le frequenze che si trovano con i settaggi in input
    mask = np.logical_and(lower <= k_freq, k_freq < upper) # trova all'interno dell'array k_freq tutti gli indici la cui frequenza è compresa tra lower e upper
    return k[mask] # ritorna un array con solo quegli indici

def compute_spec_log (Y, sr, N): #output array di 128 elementi, ognuna con la media delle frequenze per ogni nota midi dalla posizione 21 alla 108
    Y_LF = np.zeros(128)
    for p in range (21,109):
        k = pool_pitch(p, sr, N)
        if(k.shape[0] > 0):
            Y_LF[p] = Y[k].mean(axis=0)
    return Y_LF

def compute_chromagram_mio(Y_LF):
    C = np.zeros(12)
    p = np.arange(128)
    for c in range(12):
        mask = (p%12) == c
        C[c] = Y_LF[mask].sum(axis=0)
    return C

def compute_chromagram_double (Y_LF):
    Y_basso = Y_LF[0:60]
    Y_alto = Y_LF[60:]

    C_basso = np.zeros(12)
    p = np.arange(60)
    for c in range (12):
        mask = (p%12) == c
        C_basso[c] = Y_basso[mask].sum(axis=0)

    C_alto = np.zeros(12)
    p = np.arange(68)
    for c in range (12):
        mask = (p%12) == c
        C_alto[c] = Y_alto[mask].sum(axis=0)
    
    return C_basso, C_alto

def somma_ott(a, b):
    c = a+b
    if (c>12):
        c = c-12
    return c

def sottrazione_ott (a,b):
    c = a-b
    if (c<1):
        c = c+12
    return c

def tipo_accordo (accordo, basso): #out: MINORE = -1 , MAGGIORE = 1, DIMINUITO = -2, AUMENTATO = 2 // SETTIMA MIN = 10, SETTIMA MAG = 11, SESTA = 9, ADD4 = 5, ADD2 = 2
    #print("Basso: ",basso,"Accordo: ",accordo)
    tipo = 0
    estensione = 0
    #CONTROLLO PRIMA LA QUINTA PER SAPERE SE E' GIUSTA
    quinta = np.asarray(np.where(accordo == somma_ott(basso, 7)))
    if(quinta.size > 0):
        accordo = np.delete(accordo,quinta[0])

        #MAGGIORE O MINORE
        minore = np.asarray(np.where(accordo == somma_ott(basso, 3)))
        maggiore = np.asarray(np.where(accordo == somma_ott(basso, 4)))
        if (minore.size > 0):
            tipo = -1 #MIN
            accordo = np.delete(accordo,minore[0])
        elif (maggiore.size > 0):
            tipo = 1 #MAG
            accordo = np.delete(accordo,maggiore[0])
    else :
        #AUMENTATO O DIMINUITO
        aumentato = np.asarray(np.where(accordo == somma_ott(basso, 8)))
        diminuito = np.asarray(np.where(accordo == somma_ott(basso, 6)))
        if (diminuito.size > 0):
            tipo = -2 #DIM
            accordo = np.delete(accordo,diminuito[0])
        elif (aumentato.size > 0):
            tipo = 2 #AUG
            accordo = np.delete(accordo,aumentato[0])

    if(quadricordo == 1): #RICONOSCE L'ESTENSIONE
        estensione = sottrazione_ott(accordo[0],basso)     
       
    return tipo, estensione

def rimuovi_tonica (accordo, basso):
    index = np.where(accordo == basso)
    v = np.delete(accordo,index)
    return v

def chord_find(l_data, L_N, l_sr): #MAIN FUNCTION
    L_X = np.fft.fft(l_data, L_N)
    L_Y = np.abs(L_X) ** 2 # TRASFORMATA CON VALORI REALI

    #CREAZIONE SPETTRO LOGARITMICO
    LF = compute_spec_log(L_Y, l_sr, L_N)

    #CREAZIONE CROMAGRAMMI
    Cb , Ca = compute_chromagram_double(LF)

    #SORTING
    note_b = np.argsort(Cb)+1
    note = np.argsort(Ca)+1

    #SEPARO BASSO E ACCORDO
    basso = note_b[11]

    accordo = [note[11], note[10], note[9], note[8]]
    accordo = rimuovi_tonica(accordo, basso)

    #CALCOLO VETTORE DI OUTPUT
    tupla = tipo_accordo(accordo, basso)
    output = [basso, tupla[0], tupla[1]]

    return output

def adjust_chord(chord): #SE CI SONO PROBLEMI DI PITCH O DI TERZA ALLORA L'ACCORDO DIVENTA (0,0,0), SE CI SONO PROBLEMI DI ESTENSIONE ALLORA L'ESTENSIONE DIVENTA 0
    if ((chord[0] < 1) or (chord[0] > 12)):
        chord[0] = chord[1] = chord[2] = 0
    if (chord[1] == 0):
        chord[0] = chord[1] = chord[2] = 0
    if ((chord[2] != 10) and (chord[2] != 11) and (chord[2] != 9) and (chord[2] != 5) and (chord[2] != 2)):
        chord[2] = 0
    return chord

def check_chord(chord): #SE (0,0,0) TORNA 0 SENNO' TORNA 1
    if ((chord[0] == 0) or (chord[1] == 0)):
        return 0
    return 1

def translate_chord(chord):
    nomi = np.array(["ERRORE","DO", "REb", "RE", "MIb", "MI", "FA", "FA#", "SOL", "LAb", "LA", "SIb", "SI"])
    traduzione = nomi[int(chord[0])]

    if(chord[1] == 2):
        traduzione = traduzione + " Aug"
    elif(chord[1] == -1):
        traduzione = traduzione + "m"
    elif(chord[1] == -2):
        if(chord[2] == 10):
            traduzione = traduzione + "∅"
        else:
            traduzione = traduzione + " dim "

    if(quadricordo == 1):
        if((chord[2] == 10) and (chord[1] != -2)):
            traduzione = traduzione + "7"
        elif(chord[2] == 11):
            traduzione = traduzione + "maj7"
        elif(chord[2] == 9):
            traduzione = traduzione + "6"
        elif(chord[2] == 5):
            traduzione = traduzione + " add4"
        elif(chord[2] == 2):
            traduzione = traduzione + " add2"

    return traduzione

def tonality(Chords):
    DO = np.array([(1,1,0), (3,-1,0), (5,-1,0), (6,1,0), (8,1,0), (10,-1,0), (12, -2, 0)])
    REb = np.array([(2,1,0), (4,-1,0), (6,-1,0), (7,1,0), (9,1,0), (11,-1,0), (1, -2, 0)])
    RE = np.array([(3,1,0), (5,-1,0), (7,-1,0), (8,1,0), (10,1,0), (12,-1,0), (2, -2, 0)])
    MIb = np.array([(4,1,0), (6,-1,0), (8,-1,0), (9,1,0), (11,1,0), (1,-1,0), (3, -2, 0)])
    MI = np.array([(5,1,0), (7,-1,0), (9,-1,0), (10,1,0), (12,1,0), (2,-1,0), (4, -2, 0)])
    FA = np.array([(6,1,0), (8,-1,0), (10,-1,0), (11,1,0), (1,1,0), (3,-1,0), (5, -2, 0)])
    SOLb = np.array([(7,1,0), (9,-1,0), (11,-1,0), (12,1,0), (2,1,0), (4,-1,0), (6, -2, 0)])
    SOL = np.array([(8,1,0), (10,-1,0), (12,-1,0), (1,1,0), (3,1,0), (5,-1,0), (7, -2, 0)])
    LAb = np.array([(9,1,0), (11,-1,0), (1,-1,0), (2,1,0), (4,1,0), (6,-1,0), (8, -2, 0)])
    LA = np.array([(10,1,0), (12,-1,0), (2,-1,0), (3,1,0), (5,1,0), (7,-1,0), (9, -2, 0)])
    SIb = np.array([(11,1,0), (1,-1,0), (3,-1,0), (4,1,0), (6,1,0), (8,-1,0), (10, -2, 0)])
    SI = np.array([(12,1,0), (2,-1,0), (4,-1,0), (5,1,0), (7,1,0), (9,-1,0), (11, -2, 0)])

    matrix = np.array([DO, REb, RE, MIb, MI, FA, SOLb, SOL, LAb, LA, SIb, SI])

    j = 0
    check = 1
    counter = 0
    k = 0
    for i in range (0,12) : 
        while((k < n_accordi) and (check == 1)):
            tmp = counter
            for j in range (0,7) :
                check = 1
                if((Chords[k][0] == matrix[i][j][0]) and (Chords[k][1] == matrix[i][j][1])):
                    counter = counter + 1
                elif((j == 6) and (counter != tmp+1)):
                    check = 0
            k = k+1

        if(counter == n_accordi) :
            return i+1
    return -1

t = np.empty(n_t)
for i in range(n_t):
    t[i] = gap * i * 1000

Audio = AudioSegment.from_wav("Cmaj7_F_Dm7_G7_MARIMBA_RIVOLTI.wav")
for i in range(n_accordi*2): #SALVATAGGIO EXCERPT
    excerpt = Audio[t[i]:t[i+1]]
    path = './excerpt/excerpt' + str(i) + '.wav'
    excerpt.export(path, format="wav")

Window = np.empty((n_accordi * 2, int(s_r*gap) + 1))
l_sr = np.empty(n_accordi * 2)
for i in range(n_accordi * 2):
    path = "./excerpt/excerpt" + str(i) + ".wav"
    Window[i] , l_sr[i] = librosa.load(path, sr=s_r)
#LOAD TRASFORMA IN MONO FACENDO MEDIA PER OGNI ISTANTE

Chord = np.empty((n_accordi*2,3))
for i in range(n_accordi * 2):
    temp = chord_find(Window[i], L_N, l_sr[i])
    Chord[i] = adjust_chord(temp)

double = 0 #questo è un bool che dà la possibilità di trovare 2 accordi per ogni misura
j = 0
check = 0
finalChords = np.empty((n_accordi,3))
for i in range(n_accordi * 2):
    if(check_chord(Chord[i])):
        if(i%2 == 0):
            j = j+1
            finalChords[j-1] = Chord[i]
            print("Accordo 1 in misura ",j, ": ", translate_chord(Chord[i]))
            check = 1
        elif ((double==1) or (check!=1)):
            check = 0
            print("Accordo 2 in misura ",j, ": ", translate_chord(Chord[i]))

#CONTROLLO TONALITA'
tonalita = tonality(finalChords)
if(tonalita != -1) :
    nomi = np.array([" ","DO", "REb", "RE", "MIb", "MI", "FA", "FA#", "SOL", "LAb", "LA", "SIb", "SI"])
    print("\nLa tonalità della progressione è " + nomi[tonalita] + " MAGGIORE.")
quit()