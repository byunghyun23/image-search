import numpy as np
import tensorflow as tf
import click
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import pickle


def preprocessing(img):
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    return img


@click.command()
@click.option('--query_image_name', default='data/Abyssinian_1.jpg', help='Query image name')
@click.option('--target_image_dir', default='data', help='Target image path')
def run(query_image_name, target_image_dir):
    rank = {}

    kwargs = {'input_shape': (224, 224, 3),
              'include_top': False,
              'weights': 'imagenet',
              'pooling': 'avg'}
    pretrained_model = tf.keras.applications.InceptionV3(**kwargs)

    query = Image.open(query_image_name)
    query.load()

    query = preprocessing(query)
    if query.shape[3] == 1:
        raise Exception('Expected axis -1 of input shape to have value 3.')

    query_feature = pretrained_model.predict(query)

    target_images = list(Path(target_image_dir).glob(r'**/*.jpg'))
    for target_image in target_images:
        target = Image.open(str(target_image))
        target.load()

        target = preprocessing(target)
        if target.shape[3] != 3:
            continue
        target_feature = pretrained_model.predict(target)

        dist = np.linalg.norm(query_feature - target_feature)
        rank[str(target_image)] = dist

    rank = dict(sorted(rank.items(), key=lambda x: x[1]))
    print(rank)

    with open('rank.pkl', 'wb') as f:
        pickle.dump(rank, f)

    # rank = {}
    # try:
    #     with open('rank.pkl', 'rb') as f:
    #         rank = pickle.load(f)
    # except Exception as e:
    #     pass
    # print(rank)

    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(15, 7),
                             subplot_kw={'xticks': [], 'yticks': []})

    it = iter(rank)
    for i, ax in enumerate(axes.flat):
        image_name = next(it)
        print(image_name, rank[image_name])
        ax.imshow(plt.imread(image_name))
    plt.tight_layout(pad=0.5)
    plt.show()


if __name__ == '__main__':
    run()