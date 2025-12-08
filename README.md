# CS407Project
Repo for CCSU CS407 FInal Project

Environment setup: Yolo installed via ultralytics on Raspberry Pi 4, Yolo training was conducted on a seperate computer and the weights located in train/weights were transffered to the Pi for inference. 
                  Ros2 was installed on the pi using the ros-kilted-desktop distribution.

Dataset: The dataset consists of training images and validation images take from frames of the video that were split using FFMpeg. For training there are about 700 images and 400 images for validation by the model.
An interesting side effect of splitting the training and validation data this way is that the validation data contains much less examples of vehicles compared to the training data. This gives the model a harder time detcting and identifying vehicles. To address this another training run will be done to bolster the validation data.

Inference Test: The weights of the training run were trasnferred to the pi where inference was run on the validation data to test if the model was correctly predicting the classes. The data gathered from this test indicates the model steadily improved it's predictions. Bounding box accuracy improved over the course of the run which indicates the model was drawing more accurate bounding boxes (1.67 -> 0.75). Class_loss also improved a steady rate over the course of the run. It improved form 3.20 to 0.46, indicating that the model got much better at predicting the right class. Dfl_loss saw the smallest gains during the inference. Improving from 1.46 to 0.98. This indicates the model was weakest at fine tuning its bounding boxes. The data is saved in the runs directory.

