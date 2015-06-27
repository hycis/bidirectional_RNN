# bidirectional_RNN
bidirectional lstm

This repo demonstrates how to use keras to build a deep bidirectional LSTM with mlp layers before and after the LSTM layers

```python
model = Sequential()
model.add(Embedding(max_features, word_vec_len))
# transform from three dimensional to two dimensional in order to feed mlp
model.add(Transform((word_vec_len,), input=T.tensor3()))

# MLP layers
model.add(Dense(word_vec_len, 200, activation='relu'))
model.add(BatchNormalization((200,)))
model.add(Dense(200,200,activation='relu'))
model.add(BatchNormalization((200,)))
model.add(Dense(200, word_vec_len, activation='relu'))

# tranform from 2d data to 3d data again to feed RNN
model.add(Transform((maxseqlen, word_vec_len)))

# Stacked up BiDirectionLSTM layers
model.add(BiDirectionLSTM(word_vec_len, 100, output_mode='concat'))
model.add(BiDirectionLSTM(200, 24, output_mode='sum'))

# MLP layers
model.add(Reshape(24 * maxseqlen))
model.add(BatchNormalization((24 * maxseqlen,)))
model.add(Dense(24 * maxseqlen, 100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, 1, activation='sigmoid'))
```

This repo can be used for the deep speech paper

Deep Speech: Scaling up end-to-end speech recognition
arXiv:1412.5567, 2014 
A. Hannun etc
