from model import ssd
from toolkit import BBoxUtility

import numpy as np
from import_keras import image, preprocess_input

from imageio import imread
import pickle
import matplotlib.pyplot as plt
from pylab import mpl

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['font.sans-serif'] = ['SimHei']


# classes_name = ['x','p','t']
# NUM_CLASSES = 3+1
classes_name = pickle.load(open('convert/classes.pkl', 'rb'))
NUM_CLASSES = pickle.load(open('convert/classes_num.pkl', 'rb')) + 1

# classes_name = "你我他是"
# NUM_CLASSES = 4+1
input_shape = (300, 300, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

model = ssd(input_shape, NUM_CLASSES)
model.load_weights('./saved/weights.05-2.25.hdf5', by_name=True)
# model.load_weights('weights_SSD300.hdf5', by_name=True)


path_prefix = 'data/train/'  # path to your data
# path_prefix = 'data/img/'  # path to your data
gt = pickle.load(open('train.pkl', 'rb'), encoding='iso-8859-1')  # for python3.x
# gt = pickle.load(open('data_convert/train.pkl', 'rb'))
keys = sorted(gt.keys())

inputs = []
images = []
img_path = path_prefix + sorted(keys)[0]
# img_path = path_prefix + sorted(train_keys)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path).astype('float32'))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))

# %%

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

# %%

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]  # TODO

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        #         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, classes_name[label-1])
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()
