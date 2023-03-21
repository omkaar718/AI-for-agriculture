# Visual-analytics-in-agriculture

<div align="center">
    <a href="./">
        <img src="./figure/output.png" width="100%"/>
    </a>
</div>


### Usage

Inference:
```
python --source path_to_test_image --view-img --weights path_to_yolo_model_weights(runs/train/exp2/weights/best.pt) --iou-thres 0.7 --conf-thres 0.25