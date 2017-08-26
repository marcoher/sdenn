import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

data = np.loadtxt('series.out', delimiter=',')

dt_data = data[1:, 0] - data[:-1, 0]
X_data = data[:-1, 1:]
dX_data = data[1:, 1:] - data[:-1, 1:]

np.random.seed(1)

perm = np.random.permutation(len(dt_data))

train_ratio = 0.9

train_idx, test_idx = perm[:int( train_ratio*len(dt_data) )], perm[int( train_ratio*len(dt_data) ):]

X_train, X_test = X_data[train_idx], X_data[test_idx]

dX_train, dX_test = dX_data[train_idx], dX_data[test_idx]

dt_train, dt_test = dt_data[train_idx], dt_data[test_idx]

N_train = np.shape(X_train)[0]

N_test = np.shape(X_test)[0]

x_dim = np.shape(X_train)[1]

# Meta parameter: noise dimension
noise_dim = 1

# Parameters for the neural network
n_hidden0 = 10
n_hidden1 = 10
epsilon = 1e-4

# Parameters for training
n_epochs = 1000
batch_size = 1000
n_batches = N_train // batch_size
learning_rate = 1e-4

print("Train dataset of size", N_train, "split into",
      n_batches, "batches of size", batch_size)

#input("Press enter to continue...")

# Placeholders
X = tf.placeholder(tf.float32, shape=(None, x_dim), name="X")

dX = tf.placeholder(tf.float32, shape=(None, x_dim), name="dX")

dt = tf.placeholder(tf.float32, shape=(None, 1), name="dt")

keep_prob = tf.placeholder(tf.float32)

# Generic layer for neural network
def nn_layer(X,
             shape_out,
             name=None,
             activation=None,
             keep_prob=1.0,
             batch_normalized=True):

  with tf.name_scope(name):
    n_in = int(X.get_shape()[1])

    stddev = 2.0 / np.sqrt(n_in)

    W_init = tf.truncated_normal((n_in,) + shape_out, stddev=stddev)

    W = tf.Variable(W_init, name="weights")

    b = tf.Variable(tf.zeros(shape_out), name="biases")

    z = tf.nn.dropout(tf.tensordot(X, W, axes=[[-1],[0]])+b, keep_prob)

    if activation=="relu":
      if batch_normalized:
        batch_mean, batch_var = tf.nn.moments(z,[0])

        scale = tf.Variable(tf.ones(shape_out))

        beta = tf.Variable(tf.ones(shape_out))

        z_hat = tf.nn.batch_normalization(z,
                                            batch_mean,
                                            batch_var,
                                            beta,
                                            scale,
                                            epsilon)
        return tf.nn.relu(z_hat)
      else:
        return tf.nn.relu(z)
    else:
      return z

# Neural network
with tf.name_scope("dnn"):
  hidden0 = nn_layer(X,
                     (n_hidden0,),
                     "hidden0",
                     activation="relu",
                     keep_prob=keep_prob)

  hidden1 = nn_layer(hidden0,
                     (n_hidden1,),
                     "hidden1",
                     activation="relu",
                     keep_prob=keep_prob)

  drift = nn_layer(hidden1, (x_dim,), name="drift")

  sig = nn_layer(hidden1, (x_dim,), "sig")

  #vol = nn_layer(hidden1, (x_dim, noise_dim), "vol")

  #cov = tf.matmul(vol, vol, transpose_b=True)

  #dist = tf.contrib.distributions.MultivariateNormalFullCovariance(drift*tf.sqrt(dt), cov)


# Negative log likelihood
with tf.name_scope("loss"):
  root_dt = tf.sqrt(dt)
  norm_sig2 = tf.norm(sig)**2
  #dsig = tf.gradients(sig, X)
  mu = tf.matmul(dX/root_dt - drift*root_dt, sig, transpose_a=True)
  loss = tf.reduce_mean(tf.log(norm_sig2) + (mu/norm_sig2)**2)
  #loss = tf.reduce_mean(- dist.log_prob(dX/tf.sqrt(dt)))


# Training step
with tf.name_scope("train"):
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  training_op = optimizer.minimize(loss)

# Variable initializer and saver
init = tf.global_variables_initializer()

# Moving between batches
X_train_batched = X_train[:n_batches*batch_size].reshape((n_batches,batch_size,-1))

dX_train_batched = dX_train[:n_batches*batch_size].reshape((n_batches,batch_size,-1))

dt_train_batched = dt_train[:n_batches*batch_size].reshape((n_batches,batch_size,-1))

def next_batch(batch_index):
  return X_train_batched[batch_index], dX_train_batched[batch_index], dt_train_batched[batch_index]

loss_path = np.zeros(n_epochs)

# Main iteration
with tf.Session() as sess:
  init.run()

  for epoch in range(n_epochs):
    for batch_index in range(n_batches):
      X_batch, dX_batch, dt_batch = next_batch(batch_index)
      sess.run(training_op, feed_dict={X: X_batch,
                                      dX: dX_batch,
                                      dt: dt_batch,
                                      keep_prob: 0.8})

    drift_pred, sig_pred, current_loss = sess.run([drift, sig, loss],
                                                 feed_dict={X: X_test,
                                                           dX: dX_test,
                                                           dt: dt_test.reshape((-1,1)),
                                                           keep_prob: 1.0})

    loss_path[epoch] = current_loss

    plt.subplot(2,1,1).cla()
    plt.plot(loss_path[:epoch+1],'r')
    #plt.axhline(y=true_loss, color='b',linestyle='--')
    plt.xlabel("Epoch")
    plt.yscale('symlog')
    plt.ylabel("Negative log likelihood")

    plt.subplot(2,1,2).cla()
    plt.plot(X_test[:,0], sig_pred[:,0], 'b.',
             X_test[:,1], sig_pred[:,1], 'r.')
    #plt.xlabel("s")
    #plt.ylabel("$\mu(s)$")
    #plt.legend(["True","Trained"])

    plt.draw()
    plt.pause(0.0001)
    if epoch%10==1:
      print 'Completed epoch', epoch

  print("Training complete.")
  input("Press enter to exit.")
