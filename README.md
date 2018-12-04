# chinese-text-classification-pytorch
A PyTorch Implementation of Chinese Text Classification.

## Requirement
* python3
* pytorch >= 0.4
Follow setup steps in https://pytorch.org/
* jieba
* numpy
* pandas

## Usage
* Step1: Put train and test data to `./data/` folder.
* Step2: Adjust hyper parameters in `settings.ini` if necessary.
* Step3: Generate vocabulary file to the `./results/` folder.
```
python main.py --mode preprocess
```
* Step4: Train model.
Model will be saved in `./models/` folders
```
python main.py --mode train
```
* Step5: Predict labels with saved model.
`epoch_idx` is the saved model's epoch id
labels will be saved in `./results/` folder
```
python main.py --mode predict --epoch-idx 10
```
