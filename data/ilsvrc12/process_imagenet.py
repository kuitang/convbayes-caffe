__author__ = 'Sameer Lal'
import numpy as np
import sys
import caffe
import wget
import os
import time

def pre_process(limit,net,classes):
    syn_ids=[]
    
    #getting synsets
    with open('synset_words.txt') as f:
	for line in f:
		comps=line.rstrip('\n').split()
		synset=comps[0]
		if len(synset)>0:
			syn_ids.append(synset)

    #for each synset, for each image convert to weights
    j=0
    last=time.time()
    for synset in syn_ids:
	if j<classes:
		if j%10==0:
			print "Finished "+str(j)+" took: "+str(time.time()-last)
			last=time.time()
		j+=1
		k=0
		for image in os.listdir('unzipped/'+synet+'/'):
			if k<limit:
				try: 
					input_image = caffe.io.load_image('unzipped/'+image)
                                except:
                                        #os.remove(filename)
                                        valid=False
                                if valid:
                                        prediction = net.predict([input_image])

                                        #have to move file to correct folder
                                        os.rename('example.txt',"unzipped/"+str(synset)+'/'+image[:-5]+'.weights')
                                        #delete image
                                        			  

def initalize():
    caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples

    sys.path.insert(0, caffe_root + 'python')
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '../../models/bvlc_reference_caffenet/deploy.prototxt'
    PRETRAINED = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    return net


net=initalize()
start=time.time()
pre_process(10,net,1)

print 'Finished took: '+str(time.time()-start)
