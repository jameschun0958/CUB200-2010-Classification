# CUB-200-2010 Classification

* [Source url](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07)

# Introduction
This is the HW1 of 2021 NYCU VRDL courses. Train your convolution neural network to classify the bird images into the correct specifies.

# Configuration of experiment
Create a `params.json` file under `./experiments/baseline/resnet50/` folder

# Training
To train the model, run this command.
```
python3 main.py --model_dir <path-to-model-config-file>
```

# Evaluation
To evaluate model and generate `answer.txt`, run:
```
python3 test.py
```

# Result
My model achieves the following performance on :

| Model   | Top-1 Accuracy |
| -------- | -------- |
| ResNet50 | 0.664688     |
