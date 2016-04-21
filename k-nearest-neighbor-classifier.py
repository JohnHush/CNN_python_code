
def unpickle(file):
    import cPickle
    fo = open( file , 'rb' )
    dict = cPickle.load(fo)
    fo.close()
    return dict

dict1 = unpickle( './cifar-10-batches-py/data_batch_1' )
