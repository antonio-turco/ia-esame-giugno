#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
get_ipython().system('pip install -r requirements.txt')


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


train.head()


# In[4]:


train.describe()


# In[5]:


train.info()


# In[6]:


train.tail()


# In[7]:


train.select_dtypes(exclude=["number"]).describe()


# # Prime analisi e pulizia dati
# Da una prima analisi si osserva come le colonne `PatientId` e `AppointmentId` sono delle chiavi in un ipotetica relazione M-N tra paziente e visita medica, quindi si possono scartare visto che non altrerebbero il risultato. 
# 
# Inoltre si nota che nelle date non è necessario mantenere l'ora visto che l'appuntamento effettivo non mantiene questa informazione. Inoltre si scartano gli orari poiché in alcuni casi sembra che venga registrato prima l'appuntamento e poi la prenotazione. 
# Infatti osservando l'output seguente si vede che lo `scheduledDay` viene DOPO l'`appointmentDay`. Probabilmente è un informazione mancante o siccome avviene nello stesso giorno probabilmente l'orario è uguale per entrambi. 
# In ogni caso, si preferisce eliminare l'orario perché non aggiunge contenuto informativo.

# In[8]:


train.loc[70731][["ScheduledDay", "AppointmentDay"]]


# Inoltre si aggiunge una colonna chiamata `waitDays` che conterrà un valore numerico che indica i giorni che si è atteso tra la prenotazione e l'appuntamento effettivo

# In[9]:


train.ScheduledDay = pd.to_datetime(train['ScheduledDay']).dt.date
train.AppointmentDay = pd.to_datetime(train['AppointmentDay']).dt.date
train["WaitDays"] = train.AppointmentDay - train.ScheduledDay
train["WaitDays"] = train["WaitDays"].dt.days

train = train.drop(labels=["AppointmentID", "PatientId"], axis=1)


# Inoltre si nota che alcune righe hanno valori negativi di waitDays. Probabilmente sono state invertite la data di registrazione dell'appuntamento con quella della visita effettiva. Per risolvere basta cambiare il segno in `waitDays`

# In[10]:


train[train.WaitDays < 0 ].WaitDays


# In[11]:


invert_negative = lambda x: -x if x < 0 else x
train.WaitDays = train.WaitDays.apply(invert_negative)


# Applico gli stessi passaggi sul test set in modo tale da evitare disallineamenti

# In[12]:


test.ScheduledDay = pd.to_datetime(test['ScheduledDay']).dt.date
test.AppointmentDay = pd.to_datetime(test['AppointmentDay']).dt.date

test["WaitDays"] = test.AppointmentDay - test.ScheduledDay
test["WaitDays"] = test["WaitDays"].dt.days

test.WaitDays = test.WaitDays.apply(invert_negative)
test = test.drop(labels=["AppointmentID", "PatientId"], axis=1)


# ## Ricerca possibili valori nulli
# Analizzando i campi numerici non è emerso nessun dato nullo. L'unico punto ancora da analizzare sono i campi categorici che potrebbero nascondere qualche nullo sotto forma di stringa.
# I campi `NoShow` e `Gender` come si è visto prima non hanno valori nulli poiché nel conteggio delle istanze sono emersi solo due valori, che sono sono "Si/No" per `NoShow` e "M/F" per `Gender`. Quindi resta solo da valutare il campo `Neighbourhood` che come si può vedere qui sotto ha tutti valori concreti, ovvero non c'è nessun campo che indica la presenza di un valore nullo

# In[13]:


train.Neighbourhood.unique()


# ## Grafici
# In questa sezione si vuole provare ad individuare alcune possibili relazioni all'interno del dataset. In modo tale da capire meglio la realtà che si cerca di modellare e per verificiare se il modello che verrà creato utilizzerà tali osservazioni o se prenderà una strada totalmente diversa.

# In[14]:


import seaborn as sns


# Come si può osservare qui sotto, la feature `SMS_received` gioca un ruolo importante. Infatti Se si invia un sms si riesce ad aumentare la pazienza dell'utenza. Riuscendo ad avere utenti che resistono anche più di due settimane prima di dimenticare l'appuntamento

# In[15]:


sns.barplot(y="WaitDays", x="SMS_received", hue="No-show", data=train)


# Non sembra che ci siano differenze in base al genere. Infatti la forma dei grafici è molto simile. L'unica differenza è che l'utenza femminile è molto più grande di quella maschile. 

# In[16]:


sns.countplot(x="Gender", hue="No-show", data=train)


# Nell'area in basso a sinistra si può notare come le persone più giovani tendono ad avere meno pazienza di quelle più anziane. Infatti applicando gli stesso valori di `WaitDays` si osservano più valori 'Yes' nel campo `No-Show`.
# Inoltre da questo grafico emerge il fatto che chi riceve più tardi le cure di solito sono le persone sopra i 40 anni. Infatti ci sono molti casi di ritardi pari anche a più di 125 giorni.

# In[17]:


sns.scatterplot(y="WaitDays", x="Age", hue="No-show", data=train)


# Dai grafici sembra che chi ha uno `Scholarship` tenda o a dimenticarsi più spesso degli appuntamenti o ad avere meno pazienza ai ritardi. Inoltre chi ha tale servizio riesce sempre ad avere valori di `WaitDays` più piccoli di chi non attiva questo servizio

# In[18]:


sns.catplot(x="No-show", y="Scholarship", kind="bar", data=train)

plot_train = train.copy()
plot_train.Scholarship = plot_train.Scholarship.map({1:"Yes", 0:"No"})
sns.catplot(y="WaitDays", x="Scholarship", hue="No-show", data=plot_train)


# # Modellazione
# Dopo aver analizzato un minimo i dati ed effettuate alcune pulizie di base, si proverà a definire un modello che predica nuove casistiche

# In[19]:


train_raw = train.copy()
test_raw = test.copy()
train_raw.shape, test_raw.shape

y_train = train_raw["No-show"].map({"Yes":1, "No":0})
X_train = train_raw.drop(labels=["No-show"], axis=1)

y_test = test_raw["No-show"].map({"Yes":1, "No":0})
X_test = test_raw.drop(labels=["No-show"], axis=1)


# ### Label Encoding

# In[20]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# Uso l'encoder sia su `Gender` che su `Neighbourhood` visto che sono categoriche. Su `Gender` il test set dovrebbe presentare le stesse due categorie.
# Invece se su `Neighbourhood` si presentano nuove categorie l'encoder non saprebbe come gestirle e darebbe errore, la probabilità però è talmente bassa visto che il dataset è molto grande che si preferisce evitare di fare `fit_transform` anche sui dati di test. 
# Infatti facendo in tale maniera si verificherebbe un *peaking*, perché durante l'addestramento del modello si utilizzerebbero i dati del test set.
# Inoltre se si verificasse l'errore in produzione basterebbe far ripartire il training togliendo una sola riga dal test set e spostandola nel training set (quella con il nuovo valore categorico), così da evitare di dover etichettare la nuova categoria come *Unknown* e quindi senza perdere contenuto informativo. 

# In[21]:


categorical_cols= ['Gender', 'Neighbourhood']
for col in categorical_cols:
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col]) 


# Non uso l'encoder sui campi che contengono date, questo perché è molto probabile che nel test set ci siano date non presenti nel train set. Quindi si prende una data come riferimento (es. la data più piccola del train set) e la si sottrae al dataset. In questa maniera si ottengono quanti giorni (di tipo intero) sono passati da quella data in maniera compatta. Alternativamente si poteva spezzare la data nelle sue componenti giorno-mese-anno. Però così si sarebbe aggiunte 4 nuove feature che avrebbero appesantito il modello

# In[22]:


date_cols = ['ScheduledDay', 'AppointmentDay']
min_day = X_train.ScheduledDay.min()
    
for col in date_cols:
    X_train[col] = (X_train[col] - min_day).dt.days
    X_test[col] =  (X_test[col] - min_day).dt.days


# ### Training
# provo un insieme di classificatori così da scegliere quale utilizzerà per fare ottimizzazione degli iper-parametri

# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

answer_to_life_universe_everything=42

models = [ 
            ('DecisionTree', DecisionTreeClassifier(random_state=answer_to_life_universe_everything)), 
            ('LogisticRegression', LogisticRegression(random_state=answer_to_life_universe_everything)),
            ('Knn', KNeighborsClassifier())
         ]


# In[24]:


from sklearn.model_selection import cross_val_score

for name, model in models:
    print(name, 'scored:', cross_val_score(estimator=model, X=X_train, y=y_train, cv=5, n_jobs=4).mean())


# ## Calibrazione parametri
# A questo punto si utilizzerà come modello di base la `LogisticRegression`. Infatti è quella che ha dato il risultato migliore sulla cross-validation applicata al train set.
# Per ottimizzare ancora di più la predizione si cercherà di sistemare il valore di alcuni dei parametri di ingresso. Per fare questo si applicherà nuovamente la cross-validation e si sceglieranno i parametri migliori in base ai risultati ottenuti

# In[25]:


#definizione delle varie combinazioni di parametri da applicare
regularizations = [0.01, 0.1, 1, 10, 100]
intercepts = [True, False]
iterations = [20, 50, 100, 300, 500]
tolerances = [1e-5, 1e-4, 1e-2, 1]

from itertools import product
parameters = product(regularizations, intercepts, iterations, tolerances)


# In[26]:


#cross-validation su tutte le combinazioni dei parametri
scores = []
for parameter in parameters:
    regularization, intercept, iteration, tolerance = parameter
    model = LogisticRegression(
        C=regularization, 
        fit_intercept=intercept, 
        max_iter=iteration,
        tol=tolerance,
        random_state=answer_to_life_universe_everything
    )
    score = cross_val_score(
        estimator=model, 
        X=X_train, 
        y=y_train,
        cv=5,
        n_jobs=4
    ).mean()
    scores.append((regularization, intercept, iteration, tolerance, score))


# In[27]:


scores[:5]


# # Analisi parametri
# Sembra che ci siano poche differenze tra i vari test effettuati. Comunque verrà effettuta un analisi per vedere quale scelta sarà la migliore

# In[28]:


scores_data = pd.DataFrame(scores, columns=["C", "fit_intercept", "max_iter", "tol", "score"])
scores_data


# Sembra che le migliori scelte dei parametri siano praticamente identiche ai parametri di default

# In[29]:


scores_data[scores_data.score >= scores_data.score.max()]


# Il valore migliore per C sembra essere 0.1, infatti in corrispondenza di tale valore si ottiene il valore massimo nello score

# In[30]:


sns.lineplot(x="C", y="score", data=scores_data)


# In questo caso da 100 iterazioni in poi qualsiasi valore è equivalente. Si sceglie il valore 100 così da rendere più efficiente l'esecuzione del processo di learning. Anche se alla fine il fitting nel caso peggiore richiede una decina di secondi. Quindi non è ancora necessario fare ottimizzazioni più "estreme", visto che si rischia di degradare le performance in produzione

# In[31]:


sns.lineplot(x="max_iter", y="score", data=scores_data)


# i valori di tolerance applicati non cambiano in alcun modo lo score finale. Si potrebbe valutare di prendere un valore alto per avere un training veloce, però visto che su questo dataset il tempo di esecuzione del fit non è così alto alla fine non influisce in maniera significativa neanche sull'efficienza del processo di learning

# In[32]:


sns.lineplot(x="tol", y="score", data=scores_data)


# Lasciare all'algoritmo di learning di scegliere il parametro dell'intercetta sembra garantire risultati più soddisfacenti

# In[33]:


sns.catplot(x="fit_intercept", y="score", kind="box", data=scores_data);


# # Test finale
# Si conclude la realizzazione del modello. A questo punto non si applica la cross-validation sul training set in modo tale da poter utilizzare l'intero training set per l'addestramento del modello. Infine si calcola lo score sul test set

# In[34]:


final_model = LogisticRegression(
    C=0.1,
    fit_intercept=True,
    max_iter = 100,
    tol = 1e-4,
    random_state=answer_to_life_universe_everything
).fit(X_train, y_train)
final_model.score(X_test, y_test)


# # Conclusioni
# Dopo aver definito il modello finale è interessante vedere come le analisi a monte rispecchino, almeno in parte ciò che il modello ha inferito dal dataset.
# Nel processo decisionale si può notare che le feature più importanti sono:
# - Scholarship, ovvero se hanno l'assicurazione familiare
# - WaitDays, ovvero se ha aspettato molto tra prenotazione e visita
# - Età, più è alta e più è probabile che NON disdica l'appuntamento (e quindi che si presenti)
# - SMS_received, ovvero se ha ricevuto almeno un messaggio
# 
# Quindi il profilo tipico di chi salterebbe la visita è una persona con che ha l'assicurazione, che è stato notificato una o più volte via sms, che sia giovane e abbia dovuto aspettare molto per la visita effettiva.
# 
# Non sembrano essere decisive invece le feature riguardanti le malattie che il paziente può avere. Inoltre come si è visto all'inizio il genere non conta molto rispetto alla decisione finale

# In[35]:


import numpy as np
coef = np.std(X_test, 0) * final_model.coef_[0] # si prendono i coefficienti e per ottenere una stima migliore
                                                # li si moltiplicano per la deviazione standard
feature_importance = pd.Series(coef).values.reshape(1,12)
feature = pd.DataFrame(feature_importance, columns=X_train.columns)
feature

