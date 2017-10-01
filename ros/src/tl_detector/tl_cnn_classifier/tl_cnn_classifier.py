import tensorflow as tf
import numpy as np

class TLCNNClassifier(object):
    '''
    Classifer was made on base of Term 1 traffic signs clssification project
    Code is availiable at https://github.com/parilo/traffic-light-classifier
    Dataset is available at https://github.com/jorcus/CarND-Capstone-Dataset
    '''

    def __init__(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            self.sess = tf.Session(config=config)

            saver = tf.train.import_meta_graph('tl_cnn_classifier/checkpoint/cnn.ckpt.meta')
            saver.restore(self.sess, 'tl_cnn_classifier/checkpoint/cnn.ckpt')

            self.input_image = tf.get_default_graph().get_tensor_by_name("input_image:0")
            self.model_output = tf.get_default_graph().get_tensor_by_name("model_output:0")

    def get_classification(self, image):
        '''
            image 24 x 72
        '''
        with self.graph.as_default():
            classes = self.sess.run(self.model_output, {
                self.input_image: [image]
            })[0]
            return np.argmax(classes)
