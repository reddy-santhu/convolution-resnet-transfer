# Home Assignment 3

**Student Name:** Santhosh Reddy Pasnoor  
**Student ID:** 700791582  
**Course:** CS5720 Neural Network and Deep Learning  
**Semester:** Fall 2026  
**University:** University of Central Missouri  

The written answers sit in `Home Assignment 3.docx`, right after each question. This file explains how the code was run and what it produced.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 is what I used. TensorFlow does not run on 3.14.

```bash
python 01_convolution.py
python 02_transfer_learning.py
```

Printouts go in `outputs/`. The second script downloads CIFAR-10 and the ImageNet ResNet50 weights the first time it runs.

## Part I

Question 1. The input is 32×32×3, so N = 32, F = 5, S = 1, and P = 0. The output size is (32 − 5 + 0) / 1 + 1 = 28. Sixteen filters give a volume of 28×28×16. Keeping the map at 32×32 needs P = 2, because (32 − 5 + 2P) / 1 + 1 = 32 solves to P = 2.

Question 2. AlexNet showed that a deep CNN trained on GPUs, with ReLU and dropout, could win ImageNet by a wide margin. VGGNet stacks 3×3 filters. Three of them see a 7×7 neighborhood and use 27C² weights instead of 49C² for one 7×7 layer, with a ReLU after each layer. An Inception module runs 1×1, 3×3, 5×5, and pooling in parallel and concatenates them. A residual block learns F(x) and adds the input, which is how ResNet gets around the degradation problem in very deep nets.

Question 3. Representation learning means the network learns the features. Lower layers see small patches, so they pick up edges and textures. Higher layers combine those over a larger area. Transfer learning reuses a network trained on a large task. With a small image set, starting from those weights avoids fitting millions of random filters. A frozen layer does not change. A fine-tuned layer does, usually with a smaller learning rate.

## Q1: Convolution from Scratch

`01_convolution.py` stores the 5×5 input and the 3×3 filter as NumPy arrays. It slides the filter with stride 1 and padding 0 and takes a dot product at each window. The kernel is not flipped, and no library convolution is used.

The feature map is 3×3:

```
4  3  4
2  4  3
2  3  4
```

Stride 2 steps two pixels at a time, so the map shrinks to 2×2. That printout is in `outputs/q1_feature_map.txt`.

## Q2: Freeze vs Fine-Tune

`02_transfer_learning.py` loads ResNet50 with ImageNet weights, drops the original head, and adds a 2-unit softmax. The data are CIFAR-10 airplanes and automobiles. Each class has 1,500 training images, 300 validation images, and 400 test images. Train and validation do not overlap. Images are resized from 32×32 to 96×96 so the run fits on a CPU, then passed through ResNet50 preprocessing.

Both runs use 5 epochs and batch size 32. The frozen run trains only the new classifier with Adam at 1e-3. The fine-tune run also trains the ten conv5 kernels, with Adam at 1e-4. Batch-norm stays frozen so the small batches do not overwrite the ImageNet statistics.

| Method | Trainable Parameters | Training Time | Test Accuracy |
| --- | --- | --- | --- |
| Frozen Feature Extractor | 4,098 | 156.2 s | 0.9812 |
| Fine-Tuned Network | 14,957,570 | 297.5 s | 0.9875 |

The frozen training loss by epoch was 0.1067, 0.0403, 0.0230, 0.0139, 0.0094. The fine-tuned loss was 0.1066, 0.0177, 0.0059, 0.0049, 0.0003. The plot is `outputs/q2_training_loss.png`. The full table is in `outputs/q2_comparison.txt`.

Freezing is faster because the backward pass only touches 4,098 weights. Fine-tuning takes about twice as long because gradients also flow through conv5. The test gap is small, 98.12% versus 98.75%, because the ImageNet features already separate airplanes from cars. Opening conv5 still fits the training set more tightly, which is why its loss ends near 0.0003 instead of 0.0094.
