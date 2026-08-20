# YOLOv2-tiny FPGA Accelerator

## Overview

## Layer table

| #  | type    | in H×W×C     | k | stride | pad          | out H×W×C    | params     | MACs          |
|----|---------|--------------|---|--------|--------------|--------------|------------|---------------|
| 1  | conv    | 416×416×3    | 3 | 1      | 1            | 416×416×16   | 432        | 74,760,192    |
| 2  | maxpool | 416×416×16   | 2 | 2      | –            | 208×208×16   | 0          | –             |
| 3  | conv    | 208×208×16   | 3 | 1      | 1            | 208×208×32   | 4,608      | 199,360,512   |
| 4  | maxpool | 208×208×32   | 2 | 2      | –            | 104×104×32   | 0          | –             |
| 5  | conv    | 104×104×32   | 3 | 1      | 1            | 104×104×64   | 18,432     | 199,360,512   |
| 6  | maxpool | 104×104×64   | 2 | 2      | –            | 52×52×64     | 0          | –             |
| 7  | conv    | 52×52×64     | 3 | 1      | 1            | 52×52×128    | 73,728     | 199,360,512   |
| 8  | maxpool | 52×52×128    | 2 | 2      | –            | 26×26×128    | 0          | –             |
| 9  | conv    | 26×26×128    | 3 | 1      | 1            | 26×26×256    | 294,912    | 199,360,512   |
| 10 | maxpool | 26×26×256    | 2 | 2      | –            | 13×13×256    | 0          | –             |
| 11 | conv    | 13×13×256    | 3 | 1      | 1            | 13×13×512    | 1,179,648  | 199,360,512   |
| 12 | maxpool | 13×13×512    | 2 | 1      | effective 1  | 13×13×512    | 0          | –             |
| 13 | conv    | 13×13×512    | 3 | 1      | 1            | 13×13×1024   | 4,718,592  | 797,442,048   |
| 14 | conv    | 13×13×1024   | 3 | 1      | 1            | 13×13×512    | 4,718,592  | 797,442,048   |
| 15 | conv    | 13×13×512    | 1 | 1      | 1 (effective 0) | 13×13×425 | 217,600    | 36,774,400    |
| 16 | region  | 13×13×425    | – | –      | –            | detections   | –          | –             |
|    | **total** |            |   |        |              |              | **11,226,544** | **2,703,221,248** |

Params are convolution weights only; batch-norm scales/means/variances and
layer 15's biases are counted separately.

## Build

Weights (44,948,600 bytes) are not tracked in git; fetch into `models/`:

```
curl -o models/yolov2-tiny.weights https://data.pjreddie.com/files/yolov2-tiny.weights
```

## Reference model

## RTL

## Results
