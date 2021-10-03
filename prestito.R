####################################################
# linuxday 2021: Dati a chi?                       #
# Intervento   : Machine Learning con R            #
# Relatore     : Francesco Alaimo                  #
# Attribuzione : esempio: originale by Brett Lantz #
#                su analogo argomento              #
#                Corso su R: By Prof.G. Valentini  #
#                UniMI                             #
####################################################

## Identificare il rischio di concedere un prestito tramite decision tree
setwd("D:\\myCloud\\archivio\\@uninettuno\\LinuxDay 2021\\esempio di ML")
credit <- read.csv("credit.csv", stringsAsFactors = TRUE)
# step 1: esploriamo i dati di ingresso
str(credit)

# osserviamo due variabili interessanti
table(credit$checking_balance)
table(credit$savings_balance)

# alcune statistiche sui prestiti erogati
summary(credit$months_loan_duration)
summary(credit$amount)

# osserviamo la variabile default, che indica, per ogni osservazione, 
# se il cliente ha restituito il prestito o no
table(credit$default)

# usiamo una tecnica che mescola le osservazioni
RNGversion("3.5.2") # setting della versione di randomizer
set.seed(123) # il seme permette di ripetere i test mantenendo i risultati
train_sample <- sample(1000, 900)

str(train_sample)

# divide il dataframe in training e testing
credit_train <- credit[train_sample, ]
credit_test  <- credit[-train_sample, ]

# osserviamo le proporzioni tra le classi nel set di testing e di training
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

## step 2: iniziamo il training
library(C50)
credit_model <- C5.0(credit_train[-17], credit_train$default)

# mostriamo le caratteristiche del modello creato
credit_model

# mostriamo informazioni di dettaglio
summary(credit_model)

## Step 3: Valutiamo le performance del modello
# creiamo un vettore di predizioni usando il modello
credit_pred <- predict(credit_model, credit_test)

# generiamo la matrice di confusione
library(gmodels)
CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# Step 4: Miglioriamo il modello, utilizzando boosting con 10 trials
credit_boost10 <- C5.0(credit_train[-17], credit_train$default,
                       trials = 10)
credit_boost10
summary(credit_boost10)

credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
