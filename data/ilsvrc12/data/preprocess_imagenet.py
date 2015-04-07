__author__ = 'Sameer Lal'
import numpy as np
import sys
import caffe
import wget
import os
import time

def pre_process(limit,net,classes):
    for i in range(classes):
        the_class=str(i)
        image_no=0
        with open(the_class+'/imagenet_links.txt') as f:
            for line in f:
                if (image_no<limit):
                    image_no+=1
                    url=line.rstrip()
                    if(len(url)>0): 
                        valid=True
			try:
				filename=wget.download(url)
			except:
				valid=False
			if valid:
				try:
					input_image = caffe.io.load_image(filename)
                        	except:
					os.remove(filename)
					valid=False
				if valid:
					prediction = net.predict([input_image])

                       			#have to move file to correct folder
                        		os.rename('example.txt',the_class+"/"+str(image_no)+'.weights')
                        		#delete image
                        		os.remove(filename)

                        		#print prediction?


def initalize():
    caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples

    sys.path.insert(0, caffe_root + 'python')
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '../../../models/bvlc_reference_caffenet/deploy.prototxt'
    PRETRAINED = '../../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
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
