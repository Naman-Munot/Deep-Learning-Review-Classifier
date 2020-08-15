import collections
import glob
import os
import pickle
import sys
import re
import numpy
import tensorflow as tf



def GetInputFiles():
    return glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

VOCABULARY = collections.Counter()


# ** TASK 1.
def Tokenize(comment):
    """Receives a string (comment) and returns array of tokens."""
    words = re.split('[^a-z]', comment.lower())
    words = list(filter(lambda x: len(x) > 1, words))
    return words



# ** TASK 2.
def FirstLayer(net, l2_reg_val, is_training):
    """First layer of the neural network.

    Args:
        net: 2D tensor (batch-size, number of vocabulary tokens),
        l2_reg_val: float -- regularization coefficient.
        is_training: boolean tensor.A

    Returns:
        2D tensor (batch-size, 40), where 40 is the hidden dimensionality.
    """
    
    net = tf.math.l2_normalize(net, axis=1)
    net = tf.contrib.layers.fully_connected(net, 40, activation_fn=None, weights_regularizer=None, biases_initializer=None)
    net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    loss = tf.nn.l2_loss(net) * l2_reg_val
    net = tf.nn.tanh(net)
    tf.losses.add_loss(loss, tf.GraphKeys.REGULARIZATION_LOSSES)
    return net


# ** TASK 2 ** BONUS part 1
def EmbeddingL2RegularizationUpdate(embedding_variable, net_input, learn_rate, l2_reg_val):
    net_input = tf.math.l2_normalize(net_input,1)
    a=tf.transpose(net_input)
    b=tf.matmul(net_input, embedding_variable)
    derivative = tf.matmul(a,b)
    c=2*learn_rate*l2_reg_val
    derivative = tf.scalar_mul(c, derivative)
    e_var=embedding_variable - derivative
    return e_var
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    

# ** TASK 2 ** BONUS part 2
def EmbeddingL1RegularizationUpdate(embedding_variable, net_input, learn_rate, l1_reg_val):
    net_input = tf.math.l2_normalize(net_input,1)
    a=tf.transpose(net_input)
    b=tf.matmul(net_input, embedding_variable)
    c=tf.math.sign(b)
    derivative = tf.matmul(a, c)
    d=learn_rate*l1_reg_val
    derivative = tf.scalar_mul(d, derivative)
    e_var=embedding_variable - derivative
    return e_var
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    


# ** TASK 3
def SparseDropout(slice_x, keep_prob=0.5):
    """Sets random (1 - keep_prob) non-zero elements of slice_x to zero.

    Args:
        slice_x: 2D numpy array (batch_size, vocab_size)

    Returns:
        2D numpy array (batch_size, vocab_size)
    """
    a,b=numpy.nonzero(slice_x)
    drop=1-keep_prob
    c=drop*len(b)
    d=numpy.floor(c)
    inx = numpy.random.choice(len(b), int(d), replace=False)
    slice_x[a[inx],b[inx]]=0
    
    return slice_x


# ** TASK 4
# TODO(student): YOU MUST SET THIS TO GET CREDIT.
# You should set it to tf.Variable of shape (vocabulary, 40).

EMBEDDING_VAR = None



# ** TASK 5
# This is called automatically by VisualizeTSNE.
def ComputeTSNE(embedding_matrix):
    """Projects embeddings onto 2D by computing tSNE.
    
    Args:
        embedding_matrix: numpy array of size (vocabulary, 40)

    Returns:
        numpy array of size (vocabulary, 2)
    """
    embedding_mat=TSNE(n_components=2).fit_transform(embedding_matrix)
    
    return embedding_mat
    #print('visualization should generate now')
    f=plt.figure(figsize=(8,8))
    a,b,word=[],[],[]
    for a_class in class_to_words:
        if a_class=='positive':
            colors='blue'
        if a_class=='furniture':
            colors='red'
        if a_class=='location':
            colors='green'
        if a_class=='negative':
            colors='orange'
    
        for word in class_to_words[a_class]:
            index = TERM_INDEX[word]
            a = tsne_embeddings[index][0]
            b = tsne_embeddings[index][1]
            plt.scatter(a, b,marker='o',color = colors)
            plt.text(x + 0.3, y + 0.3, word, fontsize=8)
            plt.xlim(-50, 50)
            plt.ylim(-50, 50)

    plt.show()
    f.savefig("tsne_embed.pdf",bbox_inches='tight')
    #EMBEDDING_VAR = tf.get_variable    # ** TASK 4: Move and set appropriately.

    ## Build layers starting from input.
    net = x
    
    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_val)

    ## First Layer
    net = FirstLayer(net, l2_reg_val, is_training)
    
    
    EMBEDDING_VAR = tf.trainable_variables()[0]   #TASK 4


    ## Second Layer.
    net = tf.contrib.layers.fully_connected(
            net, 10, activation_fn=None, weights_regularizer=l2_reg)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.nn.relu(net)

    net = tf.contrib.layers.fully_connected(
            net, 2, activation_fn=None, weights_regularizer=l2_reg)

    return net



def main(argv):
    ######### Read dataset
    x_train, y_train, x_test, y_test = GetDataset()

    ######### Neural Network Model
    x = tf.placeholder(tf.float32, [None, x_test.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='y')
    is_training = tf.placeholder(tf.bool, [])

    l2_reg_val = 1e-6    # Co-efficient for L2 regularization (lambda)
    net = BuildInferenceNetwork(x, l2_reg_val, is_training)


    ######### Loss Function
    tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=net)

    ######### Training Algorithm
    learning_rate = tf.placeholder_with_default(
            numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def evaluate(batch_x=x_test, batch_y=y_test):
        probs = sess.run(net, {x: batch_x, is_training: False}) 
        print_f1_measures(probs, batch_y)

    def batch_step(batch_x, batch_y, lr):
            sess.run(train_op, {
                    x: batch_x,
                    y: batch_y,
                    is_training: True, learning_rate: lr,
            })

    def step(lr=0.01, batch_size=100):
        indices = numpy.random.permutation(x_train.shape[0])
        for si in range(0, x_train.shape[0], batch_size):
            se = min(si + batch_size, x_train.shape[0])
            slice_x = x_train[indices[si:se]] + 0    # + 0 to copy slice
            slice_x = SparseDropout(slice_x)
            batch_step(slice_x, y_train[indices[si:se]], lr)


    lr = 0.05
    print('Training model ... ')
    for j in range(300): step(lr)
    for j in range(300): step(lr/2)
    for j in range(300): step(lr/4)
    print('Results from training:')
    evaluate()


if __name__ == '__main__':
    tf.random.set_random_seed(0)
    main([])