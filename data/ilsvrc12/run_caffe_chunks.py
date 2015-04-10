__author__ = 'Sameer Lal'
import numpy as np
import sys
import caffe
import wget
import os
import time
import subprocess

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def pre_process(limit,net,start,end):
    syn_ids=[]

    #getting synsets
    with open('synset_words.txt') as f:
        for line in f:
                comps=line.rstrip('\n').split()
                synset=comps[0]
                if len(synset)>0:
                        syn_ids.append(synset)
    #print 'Total Classes (synsets): '+str(len(syn_ids))

    #for each synset, for each image convert to weights
    j=0
    last=time.time()
    for r in range(start,end):
        synset=syn_ids[r]
        if j%1==0 and j!=0:
        	print "Finished "+str(j)+" took: "+str(time.time()-last)
                last=time.time()
        j+=1
        subprocess.call(['sudo','mkdir','conv_weights/'+str(synset)])
        class_images=[i for i in  os.listdir('unzipped_data/'+str(synset)+'/')]
	image_batches=chunks(class_images,10)
	t=1
	#print 'Batches: '+str(len(list(image_batches)))
	for batch in image_batches:
		images=[]
                for img in batch:
                    input_image = caffe.io.load_image('unzipped_data/'+str(synset)+'/'+img)
                    images.append(input_image)
                         
                if len(images)>0:
                    prediction = net.predict([input_image],oversample=False)
                        
		#save prediction softmax to files
		np.savetxt('softmax/'+str(synset)+'_'+str(t)+".softmax", prediction)
		t+=1                        
		     
		for g in range(len(batch)):
			#have to move file to correct folder
                	img=batch[g]
			os.rename('vec_'+str(g)+'.txt',"conv_weights/"+str(synset)+'/'+img[:-5]+'.weights')



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


if __name__=="__main__":
	if len(sys.argv)!=3:
		print 'Usage: python run_caffe_chunks.py start_class end_class'
		sys.exit(1)
	start=int(sys.argv[1])
	end=int(sys.argv[2])
        net=initalize()
        begin=time.time()
        pre_process(100,net,start,end) #limit,net,start,end
	print 'Finished took: '+str(time.time()-begin)
