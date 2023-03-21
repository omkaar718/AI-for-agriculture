# Visual-analytics-in-agriculture

![output](https://user-images.githubusercontent.com/40064709/226529941-126deecb-175c-425c-8bc8-517725578223.png)

### Usage

Inference:
```
python --source path_to_test_image --view-img --weights path_to_yolo_model_weights(runs/train/exp2/weights/best.pt) --iou-thres 0.7 --conf-thres 0.25
