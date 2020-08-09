# Sentiment Analysis using voice and facial image data

In this repository, there are 2 main scripts for sentiment analysis - voice and face.

## 1. emo_recognizer_by_voice.py
I used [emoDB](http://www.emodb.bilderbar.info/start.html) and [Ravdess](https://zenodo.org/record/1188976#.Xy5vdygzaUl) for model training.
I've pushed only enodb model here.

```shell script
python3 emo_recognizer_by_voice.py <wave_file>
```

## 2. emo_recognizer_by_face.py
[OpenVINO](https://docs.openvinotoolkit.org/2019_R1/_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html) supports fast and accurate model for emotion recognition.

Check [FaceEmoRecognizer](FaceEmoRecognizer). 