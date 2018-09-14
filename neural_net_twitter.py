import numpy as n
import keras as k
import tensorflow as tf
import sys
import random

def getFullTimeline(id):
    g=open('twitter log tau.txt','at')
    t=[]
    q=[]
    nt=a.GetUserTimeline( user_id= id, include_rts=False, count =200, exclude_replies=True)
    t.extend(nt)
    o=t[-1].id-1
    print(len(t))
    while(len(nt)>0):
        nt=a.GetUserTimeline( user_id= id, include_rts=False, count =200, max_id=o, exclude_replies=True)
        t.extend(nt)
        o=t[-1].id-1
        print(len(t))
    for i in t:
        g.write("" +  str(i.text.encode('utf-8')) + " ")

with open("C:/Users/MALIXX/Documents/neural net/twitter_data.txt") as f:
    data = f.read().lower()
print('data length = {}'.format(len(data)))
ch = sorted(list(set(data)))
l=len(ch)
print('this data has {} unique features'.format(l))
ci = dict((c,i) for i, c in enumerate(ch))
ic = dict((i,c) for i, c in enumerate(ch))
file1 = open('twitter output.txt','a')
val = 60   # number of chars produced by rnn
lt = 120    # seed length
b = 4      # sequence step
seq = []
next = []
div = [0.2,0.5,0.8,1.0,1.2,1.5]

def predictor(prs,t):
    if(t==0):
        print('\ndiv zero error\n')
    prs = n.asarray(prs).astype('float64')
    exp = n.exp(n.log(prs)/t)
    prs = exp/n.sum(exp)
    p = n.random.multinomial(1,prs,1)
    return n.argmax(p)

def callback(e,log):
    print('\n----------- text generated for epoch {}'.format(e+1))
    file1.write('\n----------- text generated for epoch {}\n'.format(e))
    st = n.random.randint(0,len(data)-lt-1)
    for d in div:
        file1.write('------------ diversity = {}\n'.format(d))
        print('------------ diversity = {}'.format(d))
        g = ''
        u = data[st: st + lt]
        g+=u
        file1.write('-------------- seed: "' + g + '"\n')
        print('-------------- seed: "' + u + '"')
        sys.stdout.write(g)
        # print(type(lt))
        # print(type(l))
        for q in range(0,val):
            v = n.zeros((1,lt,l))
            for j,k in enumerate(u):
                v[0,j,ci[k]]=1.
            r = m.predict(v, verbose=0)[0]
            ni = predictor(r,d)
            nc = ic[ni]
            g+=nc
            u=u[1:]+nc
            file1.write(nc)
            sys.stdout.write(nc)
            sys.stdout.flush()
            # file1.write('\n')
        print('\n')
        file1.write('\n')

for q in range(0,len(data)-lt, b):
    seq.append(data[q: q + lt])
    next.append(data[q + lt])
seqlen=len(seq)
print('this data has {} sequences of {} chars'.format(seqlen,b))

x = n.zeros((seqlen,lt, l),dtype=n.bool)
y=n.zeros((seqlen,l),dtype=n.bool)
for i,s in enumerate(seq):
    for p,w in enumerate(s):
        x[i, p, ci[w]]=1
    y[i, ci[next[i]]]=1
outp_callback= k.callbacks.LambdaCallback(on_epoch_end=callback)
reducelr = k.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
        patience=5, min_lr=0.001)
cblist=[outp_callback, reducelr]
m=k.models.Sequential()
m.add(k.layers.LSTM(128, input_shape=(lt,l)))
# m.add(k.layers.LSTM(6, return_sequences=True))
# m.add(k.layers.Activation('tanh'))
# m.add(k.layers.Dropout(0.2))
m.add(k.layers.Dense(l))
m.add(k.layers.Activation('softmax'))
opt=k.optimizers.RMSprop(lr=0.05)
m.compile(loss='categorical_crossentropy', optimizer=opt)
m.fit(x,y,batch_size= 128,epochs=20, callbacks=cblist)
m.save_weights('twitter_model_weights_a.h5')
