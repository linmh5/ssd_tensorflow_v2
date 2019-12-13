import numpy as np
import jsonlines
import os
import pickle

epsilon = 1e-6


class Parser:

    def __init__(self, path, img_path):
        self.prefix = path
        self.img_path = img_path
        self.data = dict()
        self.classes_num = 0
        self.items = []
        self.classes = set()
        self.valid_files_num = 0

    def preprocessing(self):
        filenames = os.listdir(self.prefix)
        images = os.listdir(self.img_path)

        for file in filenames:
            with open(os.path.join(self.prefix, file), 'r') as f:
                for item in jsonlines.Reader(f):
                    if item['file_name'] in images:
                        self.valid_files_num += 1
                        self.items.append(item)
                        for sentence in item['annotations']:
                            for character in sentence:
                                self.classes.add(character['text'])
                    else:
                        print("Invalid:", item['file_name'])
        self.classes_num = len(self.classes) + 1  # TODO
        self.classes = list(self.classes)
        self.classes.append('ignore')

    def processing(self):
        self.preprocessing()

        count = 0
        for item in self.items:
            count += 1
            print("[{}/{}]".format(count, self.valid_files_num))
            width = float(item['width'])
            height = float(item['height'])
            bounding_boxes = []
            one_hot_classes = []
            for sentence in item['annotations']:
                for character in sentence:
                    x_min = float(character['adjusted_bbox'][0]) / width
                    y_min = float(character['adjusted_bbox'][1]) / height
                    # if x_min < 0:
                    #     x_min = epsilon
                    # if y_min < 0:
                    #     y_min = epsilon
                    x_max = x_min + float(character['adjusted_bbox'][2]) / width
                    y_max = y_min + float(character['adjusted_bbox'][3]) / height
                    # if x_max > 1:
                    #     x_min = 1 - epsilon
                    # if y_max > 1:
                    #     y_min = 1 - epsilon

                    bounding_boxes.append([x_min, y_min, x_max, y_max])
                    one_hot_classes.append(self._to_one_hot(character['text']))

            for ignore in item['ignore']:
                x_min = float(ignore['bbox'][0]) / width
                y_min = float(ignore['bbox'][1]) / height
                x_max = x_min + float(ignore['bbox'][2]) / width
                y_max = y_min + float(ignore['bbox'][3]) / height
                bounding_boxes.append([x_min, y_min, x_max, y_max])
                one_hot_classes.append(self._to_one_hot('ignore'))

            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))

            self.data[item['file_name']] = image_data

    def _to_one_hot(self, name):
        if name == 'ignore':
            return [0]
        else:
            return [1]
        # one_hot_vector = [0] * self.classes_num
        # try:
        #     one_hot_vector[self.classes.index(name)] = 1
        # except Exception as e:
        #     print('unknown label: %s' % name)
        # return one_hot_vector


if __name__ == '__main__':
    parser = Parser('../data/annotation/', '../data/CTW_img')
    parser.processing()
    pickle.dump(parser.data, open('train.pkl', 'wb'))
    pickle.dump(['F', 'T'], open('classes.pkl', 'wb'))
    pickle.dump(2, open('classes_num.pkl', 'wb'))

    # pickle.dump(parser.classes, open('classes.pkl', 'wb'))
    # pickle.dump(parser.classes_num, open('classes_num.pkl', 'wb'))
