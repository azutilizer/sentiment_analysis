import os
import sys
import numpy as np
import librosa
import keras

EMO_RES = ['ANGRY_EMODB', 'BOREDOM_EMODB', 'DISGUST_EMODB', 'FEAR_EMODB', 'HAPPY_EMODB', 'NEUTRAL_EMODB', 'SAD_EMODB']
RAV_RES = ['ANGRY_RAVDESS', 'CALM_RAVDESS', 'FEARFUL_RAVDESS', 'HAPPY_RAVDESS', 'NEUTRAL_RAVDESS', 'SAD_RAVDESS', 'DISGUST_RAVDESS', 'SUPRISED_RAVDESS']


def get_RAVDASS_emotion(emo_lab):
    label = int(emo_lab)
    label = label + 1
    emotion = "None"
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    if label == 1:
        emotion = "NEUTRAL_RAVDESS"
    elif label == 2:
        emotion = "CALM_RAVDESS"
    elif label == 3:
        emotion = "HAPPY_RAVDESS"
    elif label == 4:
        emotion = "SAD_RAVDESS"
    elif label == 5:
        emotion = "ANGRY_RAVDESS"
    elif label == 6:
        emotion = "FEARFUL_RAVDESS"
    elif label == 7:
        emotion = "DISGUST_RAVDESS"
    elif label == 8:
        emotion = "SUPRISED_RAVDESS"
    return emotion


def get_DUMMY_RAVDASS_emotion(emo_lab):
    label = int(emo_lab)
    label = label + 1
    emotion = "None"
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    if label == 1:
        emotion = "DU_NEUTRAL_RAVDESS"
    elif label == 2:
        emotion = "DU_CALM_RAVDESS"
    elif label == 3:
        emotion = "DU_HAPPY_RAVDESS"
    elif label == 4:
        emotion = "DU_SAD_RAVDESS"
    elif label == 5:
        emotion = "DU_ANGRY_RAVDESS"
    elif label == 6:
        emotion = "DU_FEARFUL_RAVDESS"
    elif label == 7:
        emotion = "DU_DISGUST_RAVDESS"
    elif label == 8:
        emotion = "DU_SUPRISED_RAVDESS"
    return emotion


def make_RAVDASS_prob_data(emo_prob):
    emo_prob_data = {}
    for emo_idx, emo_prob in enumerate(emo_prob):
        label = get_RAVDASS_emotion(emo_idx)
        emo_prob_data[label] = emo_prob
    return emo_prob_data


def get_EMODB_emotion(label):
    label = int(label)
    emo_label = "None"
    if label == 0:    # angry
        emo_label = "ANGRY_EMODB"
    elif label == 1:  # boredom
        emo_label = "BOREDOM_EMODB"
    elif label == 2:  # disgust
        emo_label = "DISGUST_EMODB"
    elif label == 3:  # anxiety/feat
        emo_label = "FEAR_EMODB"
    elif label == 4:  # happiness
        emo_label = "HAPPY_EMODB"
    elif label == 5:  # sadness
        emo_label = "SAD_EMODB"
    elif label == 6:  # neutral
        emo_label = "NEUTRAL_EMODB"
    return emo_label


def get_DUMMY_EMODB_emotion(label):
    label = int(label)
    emo_label = "None"
    if label == 0:    # angry
        emo_label = "DU_ANGRY_EMODB"
    elif label == 1:  # boredom
        emo_label = "DU_BOREDOM_EMODB"
    elif label == 2:  # disgust
        emo_label = "DU_DISGUST_EMODB"
    elif label == 3:  # anxiety/feat
        emo_label = "DU_FEAR_EMODB"
    elif label == 4:  # happiness
        emo_label = "DU_HAPPY_EMODB"
    elif label == 5:  # sadness
        emo_label = "DU_SAD_EMODB"
    elif label == 6:  # neutral
        emo_label = "DU_NEUTRAL_EMODB"
    return emo_label


def make_EMODB_prob_data(emo_prob):
    emo_prob_data = {}
    for emo_idx, emo_prob in enumerate(emo_prob):
        label = get_EMODB_emotion(emo_idx)
        emo_prob_data[label] = emo_prob
    return emo_prob_data


def load_model(model_dir):
    return keras.models.load_model(os.path.join(model_dir, 'Emotion_Voice_Detection_Model.h5'))


def spk_features(audio_path):
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T
    dur = len(X) / sample_rate

    return np.mean(mfcc.tolist(), axis=0), dur


def emotion_recognizer(audio_file):
    emodb_recognizer = load_model("EmoDB_model")
    feat, audio_duration = spk_features(audio_file)

    feat = np.asarray([feat])
    feat_cnn = np.expand_dims(feat, axis=2)
    emodb_duumy_label = emodb_recognizer.predict_classes(feat_cnn)
    emodb_dummy_emotion = get_DUMMY_EMODB_emotion(emodb_duumy_label)

    emodb_prob = emodb_recognizer.predict(feat_cnn)
    emodb_emotion = make_EMODB_prob_data(emodb_prob[0])

    # ravdess_recognizer = load_model("Ravdess_model")
    # ravdess_dummy_label = ravdess_recognizer.predict_classes(feat_cnn)
    # ravdes_duumy_emotion = get_DUMMY_RAVDASS_emotion(ravdess_dummy_label)
    # ravdess_prob = ravdess_recognizer.predict(feat_cnn)
    # ravdes_emotion = make_RAVDASS_prob_data(ravdess_prob[0])

    result_list = {
        "predict_dist": emodb_emotion,
        "predict_emotion": emodb_dummy_emotion,
        "duration": audio_duration
    }
    return result_list


def main(mp3_path):
    mp3_file = os.path.basename(mp3_path)
    if mp3_file.find('.wav') == -1:
        print("Invalid file format.")
        return

    res_list = emotion_recognizer(mp3_path)
    print(res_list)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("python3 emo_recognizer_by_voice.py test.wav")
        sys.exit(1)

    test_file = sys.argv[1]
    main(test_file)

