import numpy as np
import matplotlib.pyplot as plt

class NearestNeighbor( object ):
    def __init__( self ):
        pass
    
    def train( self , X , y ):
        """ X is #examples * #characters matrix , each row is an example, y is 1d of size #characters"""
        self.Xtr = X
        self.ytr = y

    def predict( self , X ):
        """ Simutaneously predict N examples arranged by rows"""
        num_test = X.shape[0]

        label_pred = np.zeros( num_test , dtype = self.ytr.dtype )

        for i in xrange( num_test ):
            if i%100 == 0:
                print i
            distances    = np.sum( np.abs(self.Xtr - X[i,:]) , axis = 1 )
            min_index    = np.argmin(distances)
            label_pred[i]= self.ytr[min_index]
        return label_pred

def unpickle(file):
    import cPickle
    fo = open( file , 'rb' )
    dict = cPickle.load(fo)
    fo.close()
    return dict

dict1 = unpickle( './cifar-10-batches-py/data_batch_1' )
dict2 = unpickle( './cifar-10-batches-py/data_batch_2' )
dict3 = unpickle( './cifar-10-batches-py/data_batch_3' )
dict4 = unpickle( './cifar-10-batches-py/data_batch_4' )
dict5 = unpickle( './cifar-10-batches-py/data_batch_5' )
dict6 = unpickle( './cifar-10-batches-py/test_batch' )

data   = np.vstack( (dict1['data'] , dict2['data'] , dict3['data']  , dict4['data']  , dict5['data'] ))
labels = np.hstack( (dict1['labels'] , dict2['labels'] , dict3['labels'] , dict4['labels'] , dict5['labels']) )
#data   = np.array(dict1['data'])
#labels = np.array(dict1['labels'])
test_data = dict6['data']
test_label= dict6['labels']

nn = NearestNeighbor()
nn.train( data , labels )
label_predicted = nn.predict(test_data)
print 'accuracy: %f' % (np.mean( test_label == label_predicted ))


#img = data[1010,:]
#img = np.reshape( img , (3,32,32) )

#img_reshape = np.zeros( (32,32,3) )
#img_reshape[:,:,0] = img[0,:,:]
#img_reshape[:,:,1] = img[1,:,:]
#img_reshape[:,:,2] = img[2,:,:]

#plt.imshow(np.uint8(img_reshape))
#plt.show()
