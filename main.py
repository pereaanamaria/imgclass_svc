import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

global flat_data_arr
global target_arr
global svc
# global param_grid


def load_categories(path):
    return os.listdir(path)


def create_dataframe(categories, data_dir):
    print(f'\nLOADING STATUS :: START')

    flat_data_arr = []
    target_arr = []

    for category in categories:
        print(f'loading... category :: {category}')
        path = os.path.join(data_dir, category)

        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(category))

        print(f'loaded category :: {category} successfully')

    print(f'LOADING STATUS :: END\n')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    data_frame = pd.DataFrame(flat_data)
    data_frame['Target'] = target
    return data_frame, flat_data_arr, target_arr


def split_data(data_frame):
    x = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]
    return train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)


def train_model(x_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.0001, 0.001, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }

    svc = svm.SVC(probability=True)
    print(f'\nTRAINING STATUS :: START')

    model = GridSearchCV(svc, param_grid, verbose=1000)
    model.fit(x_train, y_train)

    print(f'TRAINING STATUS :: END\n')

    print(f'Best parameters of the model: {model.best_params_}')
    return model


def predict_data(x_test, y_test, model):
    y_pred = model.predict(x_test)

    print(f'The predicted data is:')
    print(f'{y_pred}')

    print("The actual data is:")
    print(f'{np.array(y_test)}')

    return y_pred


def get_accuracy_percent(y_pred, y_test):
    return accuracy_score(y_pred, y_test) * 100


def dump_model(model, path):
    pickle.dump(model, open(path, 'wb'))


if __name__ == '__main__':
    # PATH = r'D:\monash\datasets\indoor\indoorCVPR_09\Images'
    # PATH = r'D:\monash\datasets\indoor\selected'
    PATH = r'D:\monash\datasets\indoor\fewer_selected'
    categories = load_categories(PATH)

    data_frame, _, _ = create_dataframe(categories, PATH)
    x_train, x_test, y_train, y_test = split_data(data_frame)
    print(f'Data was split successfully.')

    model = train_model(x_train, y_train)
    prediction = predict_data(x_test, y_test, model)
    print(f'Model accuracy: {get_accuracy_percent(prediction, y_test)}%')

    PATH_MODEL = r'img_model.p'
    dump_model(model, PATH_MODEL)
