import json

import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.models import Sequential

from toolbox.toolbox import Toolbox

toolbox = Toolbox()

binary_data_dir = "./Data/CSV/Binary/*.csv"
notes = toolbox.convert_binary2int(binary_data_dir)

# Data
n_vocab = 79

sequence_length = 100

network_input = []
network_output = []
# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append(sequence_in)
    network_output.append(sequence_out)

n_patterns = len(network_input)

# reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(n_vocab)
network_output = utils.to_categorical(network_output)

# Model
model = Sequential()
model.add(LSTM(
    128,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
# model.add(LSTM(512, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
# model.add(LSTM(256))
# model.add(Dense(256))
model.add(LSTM(64))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train
tbCallBack = TensorBoard(log_dir='./Model/Temp/GAIL/Logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         # batch_size=32,   # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True,  # 是否可视化梯度直方图
                         write_images=True,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

filepath = "./Model/Temp/GAIL/Weight/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint, TensorBoard(log_dir='./Model/Temp/GAIL/Logs')]
history = model.fit(network_input, network_output, epochs=1000, batch_size=64, callbacks=callbacks_list)

# Save History
j = json.dumps(eval(str(history.history)))

with open("./Model/Temp/GAIL/History/LSTM_history.json", "w") as f:
    f.write(j)

# Send to Wechat
sendkey = ""
title = "PyCharm"
desp = f"Model Train Finished\n\nBest Loss: {callbacks_list[0].best}"
toolbox.send_message2wechat(sendkey, title, desp)
