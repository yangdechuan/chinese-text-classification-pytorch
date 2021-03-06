# chinese-text-classification-pytorch
A PyTorch Implementation of Chinese Text Classification.

## Requirement
* python3
* pytorch >= 0.4
    * Follow setup steps in https://pytorch.org/
* jieba
* numpy
* pandas

## Usage
* Step1: Put train and test data to `./data/` folder.
* Step2: Adjust hyper parameters in `settings.ini` if necessary.
* Step3: Generate vocabulary file to the `./results/` folder.
```
python main.py --make-vocab
```
* Step4: Train model.
    * Model will be saved in `./models/` folders
```
python main.py --do-train
```
* Step5: Predict labels with saved model.
    * `epoch_idx` is the saved model's epoch id.
    * labels will be saved in `./results/` folder.
```
python main.py --do-predict --epoch-idx 10
```

## File Description
* `cnn.py` includes CNN text classifier.
* `utils.py` contains function and class regarding loading and batching data.
* `main.py` for preprocess, train or predict.
* `data/`: dataset dir
* `models/`: saved models dir
* `results/`: vocab dict file and predict result file dir
