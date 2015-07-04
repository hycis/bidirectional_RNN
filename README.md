# bidirectional_RNN
bidirectional lstm

This repo demonstrates how to use [keras](https://github.com/fchollet/keras) to build a deep bidirectional RNN/LSTM with mlp layers before and after the LSTM layers

This repo can be used for the deep speech paper from Baidu

Deep Speech: Scaling up end-to-end speech recognition
arXiv:1412.5567, 2014
A. Hannun etc

```python
max_features=20000
maxseqlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 16
word_vec_len = 256

model = Sequential()
model.add(Embedding(max_features, word_vec_len))

# MLP layers
model.add(Dense(word_vec_len, 200, activation='relu'))
model.add(BatchNormalization((200,)))
model.add(Dense(200,200,activation='relu'))
model.add(BatchNormalization((200,)))
model.add(Dense(200, word_vec_len, activation='relu'))

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
