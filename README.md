# Surgery Video Classification
This repo contains files and results for predicting the phase of a surgery based on the video images of the surgery.

Abstract: Automatic surgical phase recognition is a challenging task with the potential to to increase patient safety, reduce surgical errors and optimize the communication in the operating room. Surgical phase recognition can provide physicians with early warnings in cases of deviations and anomalies, as well as providing crucial information for archiving, educational and post-operative patient-monitoring purposes. We first attempt to address this issue by posing the problem as an image recognition task, and find that remarkably high performance is possible even without utilizing temporal features captured by sequence models such as recurrent neural networks. Our approach focuses on fine tuning pre-trained convolutional neural networks (CNNs) to perform surgical phase classification on a set of lacroscopic surgery videos. To further improve model performance and generalization, we evaluate the use of data-augmentation, balanced classes sampling, alter- nate classification loss-functions, and label-smoothing. Finally, to incorporate the temporal nature of the prob- lem, we follow the approach of Czempiel et al. (2020) by utilizing Temporal Convolutional Network (Lea et al. (2016)) in conjunction with a pre-trained ResNet50 as a visual feature extractor. With the approaches outlined above we were able to achieve a test loss of 79.5%.

# Installation
```pip install -r requirements.txt```


# Preprocessing videos
Before training, you may use the process_videos script to process videos into single jpg images for the pytorch data loader.
```python process_videos.py --input_video_folder location_to_video_folder --output_image_folder location_to_image_folder```


# To Train
The options below are possible when running train
```usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--n_workers N_WORKERS] [--n_frames N_FRAMES] [--class_resampling_weight CLASS_RESAMPLING_WEIGHT]
                [--video_file_path VIDEO_FILE_PATH] [--image_file_path IMAGE_FILE_PATH] [--save_path SAVE_PATH] [--model MODEL] [--loss LOSS] [--data_aug DATA_AUG]

Specify model and training hyperparameters.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --n_workers N_WORKERS
  --n_frames N_FRAMES - Number of frames to consider if using tcn model
  --class_resampling_weight CLASS_RESAMPLING_WEIGHT - Give 1 for no weight balancing of classes. 0 for perfect balancing. 0.5 for an average in between the two.
  --image_file_path IMAGE_FILE_PATH
  --save_path SAVE_PATH - Where to save the final model
  --model MODEL - "tcn" For temporal convolutional or default is Resnet
  --loss LOSS - "cce" for cross entropy or default is F1
  --data_aug DATA_AUG - Whether to use data augmentation or not.
```


# To Predict
Before predicting, it is required that the target videos have already been processed. A file to locate the desired video frames needs to exist at predictions.csv  with a column called "our_id" with entries like {video_name}_vi_{frame_number}. A corresponding image should exist at image_file_path/{video_name}_frame_{frame_number}.jpg

```
usage: predict.py [-h] [--batch_size BATCH_SIZE] [--n_workers N_WORKERS] [--n_frames N_FRAMES] [--image_file_path IMAGE_FILE_PATH] [--input_model_path INPUT_MODEL_PATH]
                  [--output_prediction_path OUTPUT_PREDICTION_PATH] [--model MODEL]

Specify model and training hyperparameters.

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --n_workers N_WORKERS
  --n_frames N_FRAMES
  --image_file_path IMAGE_FILE_PATH
  --input_model_path INPUT_MODEL_PATH
  --output_prediction_path OUTPUT_PREDICTION_PATH
  --model MODEL
```

# Top predictions
The top predictions achieving 0.79509% accuracy on the final kaggle competition are located at top_score_predictions_resnet18_w0.5.csv
The final model weights are located at https://drive.google.com/file/d/1mW8SFB0NVxLbbKCtC--W5G-F4i1Qla4i/view?usp=sharing