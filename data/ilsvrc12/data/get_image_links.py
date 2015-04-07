__author__ = 'Sameer Lal'
import wget
import os

#first need to download the image link for class
def get_imagenet_links(limit):
    wnid=[]
    descriptions=[]
    with open('synset_words.txt') as f:
        for line in f:
            comps=line.rstrip('\n').split()
            wnid.append(comps[0])
            descriptions.append(comps[1])

    i=0
    for wid in wnid:
        if i<limit:
                url='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='+wid
                filename=wget.download(url)
                os.makedirs(str(i)+'/')
                os.rename(filename,str(i)+'/imagenet_links.txt')
        i+=1

get_imagenet_links(1)

