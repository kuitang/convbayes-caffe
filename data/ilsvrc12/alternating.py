import numpy as np
import matplotlib.pyplot as plt
import time
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

MODEL_FILE = '../../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
img1 = 'unzipped_data/n01443537/n01443537_10014.JPEG'
img2 ='unzipped_data/n01440764/n01440764_29788.JPEG'

in1=caffe.io.load_image(img1)
in2=caffe.io.load_image(img2)

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

#images=[in1,in1,in1,in1,in1,in1,in1,in1,in1,in1]
images=[in1] * 10 + [in2] * 10
start=time.time()
prediction=net.predict(images, oversample=False)
print 'Took: '+str(time.time()-start)

print type(prediction)

np.savetxt("soft.txt", prediction)

#print 'predicted class 1:', prediction[0] 
#print 'predicted class 2:', prediction[1]
#print 'predicted class 3:', prediction[2]
#print 'predicted class 4:', prediction[3]
