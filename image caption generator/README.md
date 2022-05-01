Objective:
Create text captions for images

Model:
- EfficientnetV2 for feature extraction
- LSTM for seq2seq text generation

Process:
Images --> EfficientnetV2 to extract features --> features are fed into LSTM model to generate caption text

Data:
https://www.kaggle.com/code/xinxiawang/image-captioning-on-flickr8k-dataset
