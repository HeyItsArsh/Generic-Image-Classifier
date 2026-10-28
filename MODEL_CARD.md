# Model card

## Purpose and inputs

Seven-class single-label classification of fruit and animal photographs. Ordered outputs: Apple, Banana, Grape, Mango, Strawberry, cats, dogs. Input: 224 × 224 RGB pixels in the 0–255 range; [-1, 1] rescaling is embedded in the saved model. App/CLI/evaluation correct EXIF orientation and resize using Pillow bilinear interpolation.

## Architecture and training

Frozen ImageNet-pretrained MobileNetV2 with global-average pooled features, training-fitted normalization, dropout 0.25, and L2-regularized seven-class head (coefficient 0.002). Adam starts at 0.001; sparse categorical cross-entropy; inverse-frequency class weights. Head batch size: 64. Maximum epochs: 60; completed run: 20 with validation-loss early stopping and restored weights. Seed: 42. Training/validation/holdout sizes: 7,500/700/1,050. No backbone fine-tuning was performed.

## Recorded holdout recall

| Category | Images | Recall |
|---|---:|---:|
| Apple | 150 | 82.00% |
| Banana | 150 | 84.00% |
| Grape | 150 | 90.67% |
| Mango | 150 | 88.67% |
| Strawberry | 150 | 92.67% |
| cats | 150 | 98.67% |
| dogs | 150 | 99.33% |

Overall: 954/1,050 correct (90.86%). Validation accuracy: 90.57%. The separate existing test set scored 89.59% but has historical source overlap. Recorded comparison JSON contains both the installed-model experiment and its baseline; it is evidence from that experiment, not a fresh evaluation.

## Data audit and limitations

The source audit counted 37,030 files, 1,996 repeated files, three conflicting-label hashes, and 1,547 distinct hashes shared with the existing test collection. Exact duplicates/conflicts and existing-test hashes were excluded before selecting candidate training data. Recompressed copies or related scenes may remain; images were not perceptually grouped. Some fruit labels include desserts, drinks, or mixed scenes. A fresh external test collection is still needed.

No image dataset is distributed with this repository. Supply appropriately labeled images that you are permitted to use for training/evaluation. The source pool called test_dataset_Combined is used for training-source sampling, not as an untouched benchmark.

## Operational limits

Always chooses among seven categories. No verified unknown rejection, multiple-object localization, calibrated confidence, automatic class expansion, or sequential-retention guarantee. New training runs fit a fresh head rather than continuing the installed head. Uploading an image does not train the model. Model and label file must remain paired.

## References

- MobileNetV2: https://arxiv.org/abs/1801.04381
- TensorFlow transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning
