#%%
import numpy as np
from matplotlib import pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency
from keras.models import load_model
import pandas
import sys

# Build the VGG16 network with ImageNet weights
model = load_model('model.h5')
print('Model loaded.')
model.summary()

#%%
x_val=[]

train_data = pandas.read_csv(sys.argv[1])#training data

for i in range(train_data.shape[0]-20,train_data.shape[0]):
    temp = np.array(list(map(int, train_data.loc[i, 'feature'].split())))
    temp = temp.reshape(48,48,1)
    x_val.append(temp)

x_val = np.array(x_val)

#%%

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'conv2d_24'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
#%%
# Images corresponding to tiger, penguin, dumbbell, speedboat, spider

heatmaps = []
for idx in range(x_val.shape[0]):
    
    pred_class = np.argmax(model.predict(x_val[idx].reshape(1,48,48,1)))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], x_val[idx])
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()

