import numpy as np
import tensorflow as tf

a = np.zeros((2,2))
ta = tf.zeros((2,2))

print(a)
print(ta)
print(sess.run(ta))
#======================================================================
a = tf.constant(5.0)

b = tf.constant(6.0)

c = a*b

with tf.Session() as sess:
    print(c.eval())
#======================================================================

w = tf.Variable(tf.random_normal([5,2], stdev=0.1), name = "weight")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))
#======================================================================
a = tf.placeholder(tf.int16)

b = tf.placeholder(tf.int16)

add = tf.add(a,b)

mul = tf.multiply(a,b)

with tf.Session() as sess:
    print(sess.run(add, feed_dict = {a:2, b:3}))
    print(sess.run(mul, feed_dict = {a:2, b:3}))

sess.close()
#======================================================================
w = tf.Variable(0, dtype = tf.float32)

cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)

print(session.run(w))
# 0.0

session.run(train)

print(session.run(w))
# 0.099999994

for i in range(1000):
    session.run(train)

print(session.run(w))
# 4.9999886

session.close()
#======================================================================
coefficients = np.array([[1.], [-10.], [25.]])

w = tf.Variable(0, dtype = tf.float32)

x = tf.placeholder(tf.float32, [3,1])

cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Intialize print(session.run(w)) should be 0
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict={x: coefficients})
print(session.run(w))

# now change coefficients
coefficients = np.array([[1.], [-100.], [25.]])
# result will always be half of second term ^ (lowest point)

















s
