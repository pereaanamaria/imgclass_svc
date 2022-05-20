import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import svc_classifier

from sklearn import svm
from skimage.io import imread
from sklearn.model_selection import GridSearchCV
from skimage.transform import resize

global flat_data_arr
global target_arr


def test(path_model, categories):
    print('\nTESTING STATUS :: START')

    model = pickle.load(open(path_model, 'rb'))

    url = input('Enter URL of Image: \n')
    img = imread(url)
    plt.imshow(img)
    plt.show()

    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)

    for ind, val in enumerate(categories):
        print(f'{val} = {probability[0][ind] * 100}%')

    print(f'The predicted image is {categories[model.predict(l)[0]]}')
    print(f'Is the prediction right? (y/n)')

    while True:
        b = input()
        if b == "y" or b == "n":
            break
        print("Enter either y or n.")

    if b == 'n':
        print(f'What is the image?')
        for i in range(len(categories)):
            print(f'Enter {i} for {categories[i]}')
        k = int(input())
        while k < 0 or k >= len(categories):
            print(f'Please enter a valid number between 0-{len(categories) - 1}')
            k = int(input())

        print(f'\nLEARNING THE IMAGE STATUS :: START')

        flat_arr = flat_data_arr.copy()
        tar_arr = target_arr.copy()
        tar_arr.append(k)
        flat_arr.extend(l)
        tar_arr = np.array(tar_arr)
        flat_df = np.array(flat_arr)

        new_data_frame = pd.DataFrame(flat_df)
        new_data_frame['Target'] = tar_arr

        x_train1, x_test1, y_train1, y_test1 = svc_classifier.split_data(new_data_frame)

        new_param_grid = {}
        for param in model.best_params_:
            new_param_grid[param] = [model.best_params_[param]]

        svc = svm.SVC(probability=True)
        model1 = GridSearchCV(svc, new_param_grid, verbose=1000)
        model1.fit(x_train1, y_train1)
        y_pred1 = model1.predict(x_test1)

        print(f'New accuracy: {svc_classifier.get_accuracy_percent(y_pred1, y_test1)}%')
        svc_classifier.dump_model(model1, path_model)

        print(f'LEARNING THE IMAGE STATUS :: END\n')

    print(f'TESTING STATUS :: END\n')


def test_again():
    print(f'Do you want to test the model again? (y/n)')

    while True:
        b = input()
        if b == "y" or b == "n":
            break
        print("Enter either y or n.")

    if b == 'y':
        return True
    return False


if __name__ == '__main__':
    PATH = r'.\data\indoor'
    categories = svc_classifier.load_categories(PATH)

    data_frame, flat_data_arr, target_arr = svc_classifier.create_dataframe(categories, PATH)
    x_train, x_test, y_train, y_test = svc_classifier.split_data(data_frame)
    print(f'Data was split successfully.')

    PATH_MODEL = r'img_model.p'
    test(PATH_MODEL, categories)
    while True:
        yes = test_again()
        if yes:
            test(PATH_MODEL, categories)
        else:
            break
