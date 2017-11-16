from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
from IPython import embed
import tensorflow as tf
import numpy as np

def deepnn(x,out_dim=10):
    with tf.name_scope('reshape'):
        x_image=tf.reshape(x,[-1,28,28,1])

    with tf.name_scope('conv1'):
        W_conv1=weight_variable([5,5,1,32])
        b_conv1=weight_variable([32])
        h_conv1=tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, out_dim])
        b_fc2 = bias_variable([out_dim])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main():
    drop_prob=args.drop_prob
    mistake_rate=args.mistake
    batch_size=50
    num_label=10
    dump=open(args.file,'a')
    dump.write('#configure: vector=False,batch_size=%d,drop_prob=%.4f,mistake_rate=%.4f\n'%(batch_size,drop_prob,mistake_rate))
    ######################################################################
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = deepnn(x,out_dim=10)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,                                                                             logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(args.lr).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(batch_size)
            np_labels=batch[1].copy()
            m_inds=np.random.choice(np.arange(0,batch_size),int(batch_size*mistake_rate))
            for m_ind in m_inds:
                origin_label=batch[1][m_ind]
                origin_label=np.argmax(origin_label)
                np_labels[m_ind][origin_label]=0
                all_labels=list(range(num_label))
                all_labels.remove(origin_label)
                m_label=np.random.choice(all_labels)
                np_labels[m_ind][m_label]=1

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                dump.write('step %d, training accuracy %g\n' % (i, train_accuracy))
                print('step %d, training accuracy %g' % (i, train_accuracy))
            if i%1000==0:
                dump.write('\ni=%d,test_acc=%g\n\n' % (i,accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
                print('i=%d,test accuracy %g' % (i,accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))

            train_step.run(feed_dict={x: batch[0], y_: np_labels, keep_prob:drop_prob})
        print('Saving model...')
        saver.save(sess,'model/basic_',global_step=i)
        dump.write('\ni=%d,test_acc=%g\n\n' % (i,accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
        print('i=%d,test accuracy %g' % (i,accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))

def print_mnist(x):
    for i in range(28):
        for j in range(28):
            if x[i*28+j]>0:
                print(1,end='')
            else:
                print(0,end='')
        print()

# vector algorithm
def vector_loss():
    out_dim=args.out_dim
    num_label=10
    batch_size=50
    drop_prob=args.drop_prob
    mistake_rate=args.mistake
    if args.dynamic_lr:
        lr=args.lr*float(num_label)/float(out_dim)
    else:
        lr=args.lr
    dump=open(args.file,'a')
    dump.write('#configure: vector=True,out_dim=%d,batch_size=%d,drop_prob=%.4f,mistake_rate=%.4f,dynamic_lr=%d\n'%(out_dim,batch_size,drop_prob,mistake_rate,args.dynamic_lr))
    #################################################################
    mnist = input_data.read_data_sets(args.data_dir, one_hot=False)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, out_dim])
    y_conv, keep_prob = deepnn(x,out_dim=out_dim)
    loss=tf.nn.l2_loss(tf.random_uniform([out_dim],minval=0.1,maxval=1)*(y_conv-y_))
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver()
    num_peer=int(out_dim/num_label) # how many value does one label have
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(batch_size)
            np_labels=batch[1].copy()
            m_inds=np.random.choice(np.arange(0,batch_size),int(batch_size*mistake_rate))
            for m_ind in m_inds:
                origin_label=batch[1][m_ind]
                all_labels=list(range(num_label))
                all_labels.remove(origin_label)
                m_label=np.random.choice(all_labels)
                #embed()
                np_labels[m_ind]=m_label
            input_np_labels=np.zeros([batch_size,out_dim])
            for batch_ind in range(batch_size):
                cur_label=np_labels[batch_ind]
                input_np_labels[batch_ind,cur_label*num_peer:(cur_label+1)*num_peer]=1
            if i % 100 == 0:
                y_conv_val=sess.run(y_conv,feed_dict={
                    x: batch[0], keep_prob: 1.0})
                # get acc
                train_acc=0
                #print('label:',np_labels[0]);print('out:',y_conv_val[0]);raw_input()
                for batch_ind in range(batch_size):
                    max_feature=0
                    cur_label=0
                    for label_ind in range(num_label):
                        cur_label_sum=np.sum(y_conv_val[batch_ind,label_ind*num_peer:(label_ind+1)*(num_peer)])
                        if cur_label_sum>max_feature:
                            cur_label=label_ind
                            max_feature=cur_label_sum
                    if cur_label==batch[1][batch_ind]:
                        train_acc+=1
                print('i=%d,acc=%.4f'%(i,train_acc/batch_size))
                dump.write('i=%d,acc=%.4f\n'%(i,train_acc/batch_size))
            if i%1000==0:
                test_num=len(mnist.test.images)
                test_out=sess.run(y_conv,feed_dict={
                        x: mnist.test.images, keep_prob: 1.0})
                test_labels=mnist.test.labels
                test_acc=0
                for batch_ind in range(test_num):
                    max_feature=0
                    cur_label=0
                    for label_ind in range(num_label):
                        cur_label_sum=np.sum(test_out[batch_ind,label_ind*num_peer:(label_ind+1)*(num_peer)])
                        if cur_label_sum>max_feature:
                            cur_label=label_ind
                            max_feature=cur_label_sum
                    if cur_label==test_labels[batch_ind]:
                        test_acc+=1
                if i>15000:print('label:',test_labels[0]);print(test_out[0]);embed();exit()
                dump.write('\ni=%d,test_acc=%.4f\n\n'%(i,test_acc/test_num))
                print('i=%d,test_acc=%.4f'%(i,test_acc/test_num))
            train_step.run(feed_dict={x: batch[0], y_: input_np_labels, keep_prob: drop_prob})
        print('Saving model...')
        saver.save(sess,'model/vector_',global_step=i)
        print('preparing to calculate test acc')
        test_num=len(mnist.test.images)
        test_out=sess.run(y_conv,feed_dict={
                x: mnist.test.images, keep_prob: 1.0})
        test_labels=mnist.test.labels
        test_acc=0
        for batch_ind in range(test_num):
            max_feature=0
            cur_label=0
            for label_ind in range(num_label):
                cur_label_sum=np.sum(test_out[batch_ind,label_ind*num_peer:(label_ind+1)*(num_peer)])
                if cur_label_sum>max_feature:
                    cur_label=label_ind
                    max_feature=cur_label_sum
            if cur_label==test_labels[batch_ind]:
                test_acc+=1
        dump.write('\ntest_acc=%.4f\n\n'%(test_acc/test_num))
        dump.close()
        print('i=%d,acc=%.4f'%(i,test_acc/test_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                    default='data',
                    help='Directory for storing input data')
    parser.add_argument('-vector',action='store_true',help='use vector algorithm instead of scalar.')
    parser.add_argument('-out_dim',type=int,default=100,help='specify output dimension, note that has to be times of 10.')
    parser.add_argument('-drop_prob',type=float,default=1.,help='specify dropout probability')
    parser.add_argument('-mistake',type=float,default=0.,help='specify mistake rate')
    parser.add_argument('-gpu',type=str,default='0',help='Specify gpu to use')
    parser.add_argument('-file',type=str,default='experiment',help='specify the name to write experiment result')
    parser.add_argument('-lr',type=float,default=0.0001,help='specify learning rate')
    parser.add_argument('-dynamic_lr',action='store_true',help='specify to open dynamic learning rate mode on')
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # closure get command-line args
    if args.vector:
        vector_loss()
    else:
        main()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    #tf.app.run(main=vector_loss, argv=[sys.argv[0]] + unparsed)
