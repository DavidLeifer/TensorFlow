#kinda like euclidist, but for skulls
import tensorflow as tf
import numpy as np
import scipy
from scipy.spatial.distance import mahalanobis
import time
import itertools
#check libs
#help('modules')

#build an array
num_obs = 100
num_var = 2
data = np.zeros([num_obs,num_var])

#iterate randos
for i in range(num_obs):
	data[i][0] = np.random.randint(2)
	time.sleep(0.01)
	data[i][1] = data[i][0]*50 + np.random.randint(51)

#covariance matrix
V = np.cov(data,rowvar=0)
Vm = np.matrix(V)
x = data[0]
y = data[1]

##playing with numbers- maha dist 
#xy = x-y
#xy.transpose() * Vm.I * xy.reshape(num_var)
#xy.reshape(2,1)
#s = _
#np.sqrt(s[0][0])
#np.sqrt(s)
#m = _
#np.sqrt(m)
#mahalanobis(x,y,Vm.I)
#ma = _
#np.sqrt(ma)

#create mahalanobis function
def maha(x,y,V):
	return np.sqrt((x-y).transpose() * Vm.I * (x-y).reshape(len(x),1))

#start using tf
sess = tf.Session()

#maha tf style(x,y,Vm)
def tf_maha(x,y,VI):
	num_var = len(x)
	with tf.Session() as sess:
		in1 = tf.placeholder(tf.float32,[num_var])
		in2 = tf.placeholder(tf.float32,[num_var])
		out = tf.placeholder(tf.float32,[num_var,num_var])
		vivi = tf.matrix_inverse(out)
		diff = in1-in2
		dt = tf.reshape(tf.transpose(diff),[1,len(x)])
		ds = tf.reshape(diff,[len(x),1])
		M1 = dt * vivi
		M2 = M1 * ds 
		output = tf.sqrt(M2)
		ans = sess.run([output],feed_dict = {in1:x,
											in2:y,
											out:VI})
	return ans
#test
tf_maha(x,y,Vm.I)

#test2
tf_maha(x,y,Vm)

#compute variance covariance matrix of data
#sum([d[v1]*d[v2] for d in dif])
#dif[:,v1] * dif[:,v2]
v1 = 0
v2 = 1
def cov(data): 
	num_obs, num_vars = data.shape
	mns = np.mean(data,axis=0)
	dif = data - mns
	cov = np.zeros([num_vars,num_vars])
	for v1,v2 in itertools.combinations_with_replacement(range(num_vars),2):
		cov[v1,v2] = (1/float(num_obs-1))*np.sum(dif[:,v1] * dif[:,v2])
		cov[v2,v1] = cov[v1,v2]
	return cov

#compute variance covariance matrix in tflow
def tf_cov(data):
	num_obs, num_vars = data.shape
	with tf.Session() as sess:
		bra = tf.placeholder(tf.float32,[num_obs,num_vars])
		avg = tf.reduce_mean(bra, reduction_indices=[0])
		dif = tf.sub(bra,avg)
		cov = (1/float(num_obs-1)) * tf.reduce
		ans = sess.run([cov],feed_dict = {bra:data})
	return ans
V = np.cov(data,rowvar=0)

#compute maha distance from points to the mean
def tf_maha(data):
	num_obs, num_vars = data.shape
	with tf.Session() as sess:
		dd = tf.placeholder(tf.float64,[num_obs,num_vars])
		means = tf.reduce_mean(dd, reduction_indices=[0])
		diffs = tf.sub(dd,means)
		cov = (1/float(num_obs-1)) * tf.matmul(tf.transpose(diffs),diffs)
		vivi = tf.matrix_inverse(cov)
		dt = tf.reshape(tf.transpose(diffs),[num_obs,num_var])
		ds = tf.reshape(diffs,[num_vars,num_obs])
		M1 = tf.matmul(dt,vivi)
		M2 = tf.matmul(M1,ds)
		z = tf.pack([M2[i,i] for i in range(num_obs)])
		output = tf.sqrt(z)
		ans = sess.run([output],feed_dict = {dd:data})
	return ans

#data[0]
#data[1]
#np.mean(data)
#np.mean(data,axis=1)




