import random, pickle
from skimage import io
import matplotlib.pyplot as plt

# 'CW_Dataset/test/' is the default string for path_to_testset
def EmotionRecognition(path_to_testset, model_type):
    # load data
    results = pickle.load(open('Code/' + model_type + '_result.p', 'rb'))
    predicted = results['predicted']
    fd_name = model_type.split('_')[0]
    data = pickle.load(open('Code/' + fd_name+'_data.p','rb'))
    labels = data['y_test']
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
    ax = axes.ravel()
    for n in range(4):
        i = random.randint(1, len(predicted) - 1)
        if i < 10:
            path = path_to_testset + 'test_000' + str(i + 1) + '_aligned.jpg'
        elif i < 100:
            path = path_to_testset + 'test_00' + str(i + 1) + '_aligned.jpg'
        elif i < 1000:
            path = path_to_testset + 'test_0' + str(i + 1) + '_aligned.jpg'
        elif i < 10000:
            path = path_to_testset + 'test_' + str(i + 1) + '_aligned.jpg'
        label = labels[i]
        img = io.imread(path)
        ax[n].imshow(img)
        emo_dict = {1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness', 5: 'Sadness', 6: 'Anger', 7: 'Neutral'}
        label = emo_dict[label]
        out = emo_dict[predicted[i]]
        ax[n].set_title(f'Label: {label} \n Prediction: {out}')
        ax[n].set_axis_off()
    fig.tight_layout()
    plt.show()

    return None

if __name__ == '__main__':
    EmotionRecognition('CW_Dataset/test/','HOG_SVM')