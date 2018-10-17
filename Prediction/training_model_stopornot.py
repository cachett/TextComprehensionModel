#!/usr/bin/env python3


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
import random
from tqdm import tqdm
import numpy as np

batch_size = 128
epochs = 10
test_size = 500


out_model = 'model_checkpoint_{}_batch_{}_epochs.h5'.format(batch_size, epochs)




print("Reading input")
with open("train_500_stopornot.in","r") as f:
    train_in = f.read().split('\n')
    train_in = [eval(i) for i in tqdm(train_in[:-1])]
    print("done train in")

print("Reading output")
with open("train_500_stopornot.out","r") as f:
    train_out = f.read().split('\n')
    train_out = [eval(i) for i in tqdm(train_out[:-1])]
    print("done train out")

decision_stop = []
decision_continue = []

print("balancing data...")
for n, _ in tqdm(enumerate(train_in)):
    input_layer = train_in[n]
    output_layer = train_out[n]

    if output_layer == [1,0]:
        decision_stop.append([input_layer, output_layer])
    elif output_layer == [0,1]:
        decision_continue.append([input_layer, output_layer])

print(len(decision_stop), len(decision_continue))
shortest = min(len(decision_stop), len(decision_continue))

random.shuffle(decision_stop)
random.shuffle(decision_continue)

#Here we loose information to balance data
decision_stop = decision_stop[:shortest]
decision_continue = decision_continue[:shortest]

print(len(decision_stop), len(decision_continue))

all_choices = decision_stop + decision_continue
random.shuffle(all_choices)

train_in = []
train_out = []

print("rebuilding training data...")
for x,y in tqdm(all_choices):
    train_in.append(x)
    train_out.append(y)

np.save("train_in.npy", train_in)
np.save("train_out.npy", train_out)

train_in = np.load("train_in.npy")
train_out = np.load("train_out.npy")

print('train_in:',len(train_in))

x_train = train_in[:-test_size]
y_train = train_out[:-test_size]

x_test = train_in[-test_size:]
y_test = train_out[-test_size:]

print('Building model...')
model = Sequential()
model.add(Dense(128, input_shape=(len(train_in[0]),)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, input_shape=(128,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(len(train_out[0])))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

model.save(out_model)
print("Model saved to:",out_model)
print('Test score:', score[0])
print('Test accuracy:', score[1])
