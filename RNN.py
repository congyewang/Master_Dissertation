import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
from keras.utils import plot_model


str_network_input_command = ''
str_network_output_command = ''

for i in range(772, 787):
    exec(f'np_input_BWV{i} = pd.read_csv("./Data/CSV/Binary/BWV{i}.csv").values.T')
    exec(f'np_output_BWV{i} = pd.read_csv("./Data/CSV/Binary/BWV{i}.csv").values.T[1:, :]')
    str_network_input_command += f"np_input_BWV{i}[0:(np_input_BWV{i}.shape[0] - 1)],"
    str_network_output_command += f"np_output_BWV{i},"

network_input = eval(f"np.concatenate(({str_network_input_command}), axis=0)")
network_input = network_input[0:network_input.shape[0], :, np.newaxis]
network_output = eval(f"np.concatenate(({str_network_output_command}), axis=0)")

n_vocab = 12
model = Sequential()
model.add(SimpleRNN(
    12,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(SimpleRNN(12, return_sequences=True))
model.add(SimpleRNN(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(n_vocab, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = "./Model/Temp/RNN/Weight/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]
history = model.fit(network_input, network_output, epochs=1000, batch_size=16,
                    callbacks=[TensorBoard(log_dir='./Model/Temp/RNN/Logs')])

plot_model(model, to_file='./Pic/RNN/model.png', dpi=600)

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('./Pic/RNN/accuracy.png', dpi=600)

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('./Pic/RNN/loss.png', dpi=600)

j = json.dumps(eval(str(history.history)))

with open("./Model/Temp/RNN/History/RNN_history.json", "w") as f:
    f.write(j)

# Re-read model weights
n_vocab = 12
model = Sequential()
model.add(SimpleRNN(
    12,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(SimpleRNN(12, return_sequences=True))
model.add(SimpleRNN(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(n_vocab, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('./Model/Complete/RNN/Weight/weights-improvement-1000-0.0000-bigger.hdf5')

model.save("./Model/Complete/RNN/RNN.h5")

random.seed(1234)
note_number_list = range(2)
note_location_list = range(12)

note_number = random.sample(note_number_list, 1)
note_location = random.sample(note_number_list, (note_number[0] + 1))

init_note = np.zeros(12)
for i in note_location:
    init_note[i] = 1

init_note = init_note.reshape((1, init_note.shape[0], 1))

create_length = 100
s = np.zeros([create_length, 12])
input_note = init_note
for i in range(create_length):
    predict_note = model.predict(input_note)
    input_note = np.apply_along_axis(lambda x: 1 if x > 0.5 else 0, 0, predict_note)
    s[i] = input_note
    input_note = input_note.reshape((1, input_note.shape[0], 1))
