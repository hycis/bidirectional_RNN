# bidirectional_RNN
bidirectional lstm

This repo demonstrates how to use [keras](https://github.com/fchollet/keras) to build a deep bidirectional RNN/LSTM with mlp layers before and after the LSTM layers

This repo can be used for the deep speech paper from Baidu

Deep Speech: Scaling up end-to-end speech recognition
arXiv:1412.5567, 2014
A. Hannun etc


<!-- ![BiLSTM](images/illustration.png "Title" {width=40px height=400px}) -->
<img src="item_lstm.png" height="250">


```python
max_features=20000
maxseqlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 16
word_vec_len = 256

model = Sequential()
model.add(Embedding(max_features, word_vec_len))

# MLP layers
model.add(Transform((word_vec_len,))) # transform from 3d dimensional input to 2d input for mlp
model.add(Dense(word_vec_len, 100, activation='relu'))
model.add(BatchNormalization((100,)))
model.add(Dense(100,100,activation='relu'))
model.add(BatchNormalization((100,)))
model.add(Dense(100, word_vec_len, activation='relu'))
model.add(Transform((maxseqlen, word_vec_len))) # transform back from 2d to 3d for recurrent input

# Stacked up BiDirectionLSTM layers
model.add(BiDirectionLSTM(word_vec_len, 50, output_mode='concat', return_sequences=True))
model.add(BiDirectionLSTM(100, 24, output_mode='sum', return_sequences=True))

# MLP layers
model.add(Reshape(24 * maxseqlen))
model.add(BatchNormalization((24 * maxseqlen,)))
model.add(Dense(24 * maxseqlen, 100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, 1, activation='sigmoid'))
```
