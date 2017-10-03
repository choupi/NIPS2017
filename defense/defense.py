from inception_resnet_v2 import InceptionResNetV2
from inception_resnet_v2 import preprocess_input as Inc_preprocess_input
from xception import Xception
from resnet50 import ResNet50
from vgg19 import VGG19
from keras.preprocessing import image
import keras.applications as app
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

def list_images(input_dir):
    images = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    print(len(images), 'images')
    return images

def chunk(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

input_dir = sys.argv[1]
out_fn = sys.argv[2]
images = list_images(input_dir)
out_file = open(out_fn, 'w')

imodel = InceptionResNetV2(weights='imagenet')
xmodel = Xception(weights='imagenet')
rmodel = ResNet50(weights='imagenet')
vmodel = VGG19(weights='imagenet')

for fs in chunk(images,100):
    xs299=[]
    xs224=[]
    for f in fs:
        img_path = join(input_dir, f)
    
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = Inc_preprocess_input(x)
        xs299.append(x[0,:,:])
    
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        xs224.append(x[0,:,:])

    xs299=np.array(xs299)
    xs224=np.array(xs224)
    print xs299.shape, xs224.shape
    ipreds = imodel.predict(xs299)
    xpreds = imodel.predict(xs299)
    #print('Inc Predicted:', decode_predictions(ipreds, top=5)[0])
    rpreds = rmodel.predict(xs224)
    vpreds = vmodel.predict(xs224)
    #print('Res Predicted:', decode_predictions(rpreds, top=5)[0])
    #print('VGG Predicted:', decode_predictions(vpreds, top=5)[0])
    preds=(ipreds+xpreds+rpreds+vpreds)/4.0
    #print(preds)
    #print(f, 'Predicted:', decode_predictions(preds, top=5)[0])
    for i in xrange(preds.shape[0]):
        l=preds[i].argsort()[-1:][::-1][0]+1
        print(fs[i], l)
        out_file.write('{0},{1}\n'.format(fs[i], l))
    #print('=====================================')
