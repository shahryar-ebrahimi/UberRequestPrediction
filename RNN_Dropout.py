
# =======================================================================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import time

# ======================================================================================================================

DATA = read_csv('UberDemand.csv')
DATA = np.asarray(DATA)
DATA = DATA[:, 2:-1]
PICKSUP = DATA[:, 0]
Temp = np.zeros([4343, 18])
for i in range(int(DATA.shape[0] / 4)):
    Temp[i, 0:4] = PICKSUP[4 * i:(4 * i) + 4]
    Temp[i, 4:] = DATA[4 * i, 1:]
DF = Temp

# ******************************************************************************
# Plotting the hr_sin versus hr_cos to find a way to generate a new feature
A = DF[:, 14] # hr_sin
B = DF[:, 15] # hr_cos


C = A - .5  # hr_sin
D = B - .5  # hr_cos

plt.figure()
plt.scatter(D, C)
plt.title('Feature: Hour')

E = np.zeros([D.shape[0], ])
for i in range(D.shape[0]):
    if D[i] == 0:
        D[i] = 0.000000000000000001   # set the value equal to epsillon
        
        
for i in range(D.shape[0]):
    if C[i] < 0 and D[i] < 0:
        E[i] = np.arctan(C[i] / D[i])+np.pi
    elif D[i] < 0 and C[i] > 0:
        E[i] = np.arctan(C[i] / D[i]) + np.pi
    else:
        E[i] = np.arctan(C[i] / D[i])

hr_degree = 180 * E / (np.pi)

# ******************************************************************************
# Plotteing the day_sin versus day_cos to find a way to generate a new feature
AD = DF[:, 16]  # day_sin
BD = DF[:, 17]  # day_cos

CD = AD - .5  # day_sin
DD = BD - .5  #day_cos

plt.figure()
plt.scatter(DD, CD)
plt.title('Feature: Day')

for i in range(DD.shape[0]):
    if DD[i] == 0:
        DD[i] = 0.000000000000000001  # set the value equal to epsillon

ED = np.zeros([DD.shape[0], ])
for i in range(D.shape[0]):
    if CD[i] < 0 and DD[i] < 0:
        ED[i] = np.arctan(CD[i] / DD[i]) + np.pi
    elif DD[i] < 0 and CD[i] > 0:
        ED[i] = np.arctan(CD[i] / DD[i]) + np.pi
    else:
        ED[i] = np.arctan(CD[i] / DD[i])

day_degree = 180 * ED / (np.pi)


# ******************************************************************************
# Final data-frame
DF[:, 14] = hr_degree
DF[:, 15] = day_degree
DF = DF[:, :-2]
for i in range(DF.shape[1]):  # Normalization of the data 
    DF[:, i] /= max(DF[:, i])


DF = DF[:, [0, 1, 2, 3, 6, 7, 9, 10, 11, 14, 15]]
# =======================================================================================================================
# Splitting the data to training, validation, and testing set
TEMP =np.zeros([DF.shape[0]-6, 6, DF.shape[1]])
for i in range(DF.shape[0]-6):
    TEMP[i, :, :] = DF[i:i+6, :]

DATAFRAME = TEMP 
print(DATAFRAME.shape)


TRAINx = DATAFRAME[:3000, :, :]
TRAINy = DF[6:TRAINx.shape[0]+6, :4]

VALIDx = DATAFRAME[3000:4000, :, :]
VALIDy = DF[3000+6:4000+6, :4]


TESTx = DATAFRAME[4000:, :, :]
TESTy = DF[4000+6:DATAFRAME.shape[0]+6, :4]


EPOCH = 50  # Number of epochs
# =======================================================================================================================
# Biulding Model...
# Simple LSTM Network
MODEL1 = Sequential()
MODEL1.add(SimpleRNN(30, input_shape=(TRAINx.shape[1], TRAINx.shape[2])))
MODEL1.add(Dropout(0.2))
MODEL1.add(Dense(4))

MODEL1.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

start1 = time.clock()

result1 = MODEL1.fit(TRAINx, TRAINy, epochs=EPOCH, batch_size=5, validation_data=(TESTx, TESTy), verbose=1,
                   shuffle=False)

stop1 = time.clock()

# *****************************************************************************
# Without Dropout
MODEL3 = Sequential()
MODEL3.add(SimpleRNN(30, input_shape=(TRAINx.shape[1], TRAINx.shape[2])))
MODEL3.add(Dense(4))

MODEL3.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

start3 = time.clock()

result3 = MODEL3.fit(TRAINx, TRAINy, epochs=EPOCH, batch_size=5, validation_data=(TESTx, TESTy), verbose=1,
                   shuffle=False)

stop3 = time.clock()

# =======================================================================================================================
# ======================================================================================================================
# Pridiction..
Y1_Predict = MODEL1.predict(TESTx)
Y3_Predict = MODEL3.predict(TESTx) 

# ======================================================================================================================
# RESULTS
# Loss for training 
plt.figure()
plt.plot(result1.history['loss'], label='With Dropout')
plt.plot(result3.history['loss'], label='Without Dropout')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss for with/without Dropout for train data')
plt.show()

# *****************************************************************************

# Loss for validating
plt.figure()
plt.plot(result1.history['val_loss'], label='With Dropout')
plt.plot(result3.history['val_loss'], label='Without Dropout')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss for with/without Dropout for test data')
plt.show()

# *****************************************************************************

# Accuracy for training 
plt.figure()
plt.plot(result1.history['acc'], label='With Dropout')
plt.plot(result3.history['acc'], label='Without Dropout')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy for with/without Dropout for train data')
plt.show()

# *****************************************************************************

# Accuracy for validating
plt.figure()
plt.plot(result1.history['val_acc'], label='With Dropout')
plt.plot(result3.history['val_acc'], label='Without Dropout')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy for with/without Dropout for test data')
plt.show()

# *****************************************************************************
# Plotting Predicted value versus True value for with dropout
plt.figure()
plt.scatter(TESTy[:, 0], Y1_Predict[:, 0])
plt.ylabel('Predicted value(Bronx)')
plt.xlabel('True value')
plt.title('With Dropout: True VS Predicted Value for Test Data')
plt.show()


plt.figure()
plt.scatter(TESTy[:, 1], Y1_Predict[:, 1])
plt.ylabel('Predicted value(Brooklyn)')
plt.xlabel('True value')
plt.title('With Dropout: True VS Predicted Value for Test Data')
plt.show()


plt.figure()
plt.scatter(TESTy[:, 2], Y1_Predict[:, 2])
plt.ylabel('Predicted value(Manhattan)')
plt.xlabel('True value')
plt.title('With Dropout: True VS Predicted Value for Test Data')
plt.show()


plt.figure()
plt.scatter(TESTy[:, 3], Y1_Predict[:, 3])
plt.ylabel('Predicted value(Queens)')
plt.xlabel('True value')
plt.title('With Dropout: True VS Predicted Value for Test Data')
plt.show()

print("Training time for RNN = ", stop1-start1)


# *****************************************************************************
# Plotting Predicted value versus True value for Adam
plt.figure()

plt.scatter(TESTy[:, 0], Y3_Predict[:, 0])
plt.ylabel('Predicted value(Bronx)')
plt.xlabel('True value')
plt.title('Without Dropout: True VS Predicted Value for Test Data')
plt.show()


plt.figure()
plt.scatter(TESTy[:, 1], Y3_Predict[:, 1])
plt.ylabel('Predicted value(Brooklyn)')
plt.xlabel('True value')
plt.title('Without Dropout: True VS Predicted Value for Test Data')
plt.show()


plt.figure()
plt.scatter(TESTy[:, 2], Y3_Predict[:, 2])
plt.ylabel('Predicted value(Manhattan)')
plt.xlabel('True value')
plt.title('Without Dropout: True VS Predicted Value for Test Data')
plt.show()


plt.figure()
plt.scatter(TESTy[:, 3], Y3_Predict[:, 3])
plt.ylabel('Predicted value(Queens)')
plt.xlabel('True value')
plt.title('Without Dropout: True VS Predicted Value for Test Data')
plt.show()

print("Training time for GRU = ", stop3-start3)


# *****************************************************************************
# SGD: Plotting the predicted data for four boroughs
plt.figure()

x=np.arange(TESTy.shape[0])

plt.figure()
plt.plot(x,TESTy[:, 0],label='True')
plt.plot(x,Y1_Predict[:, 0],label='Predicted')
plt.title('With Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Bronx)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,TESTy[:, 1],label='True')
plt.plot(x,Y1_Predict[:, 1],label='Predicted')
plt.title('With Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Brooklyn)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,TESTy[:, 2],label='True')
plt.plot(x,Y1_Predict[:, 2],label='Predicted')
plt.title('With Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Manhattan)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,TESTy[:, 3],label='True')
plt.plot(x,Y1_Predict[:, 3],label='Predicted')
plt.title('With Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Queens)')
plt.legend()
plt.show()


# *****************************************************************************
# Adam: Plotting the predicted data for four boroughs
plt.figure()

x=np.arange(TESTy.shape[0])

plt.figure()
plt.plot(x,TESTy[:, 0],label='True')
plt.plot(x,Y3_Predict[:, 0],label='Predicted')
plt.title('Without Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Bronx)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,TESTy[:, 1],label='True')
plt.plot(x,Y3_Predict[:, 1],label='Predicted')
plt.title('Without Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Brooklyn)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,TESTy[:, 2],label='True')
plt.plot(x,Y3_Predict[:, 2],label='Predicted')
plt.title('Without Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Manhattan)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x,TESTy[:, 3],label='True')
plt.plot(x,Y3_Predict[:, 3],label='Predicted')
plt.title('Without Dropout: True and Predicted Value for Test Data')
plt.xlabel('Time')
plt.ylabel('Values(Queens)')
plt.legend()

plt.show()
