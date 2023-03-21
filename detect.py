import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general_debug import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.create_2D_map import four_point_transform, generate_graphics
from utils.activity_recognition import activity_recognition_run
import numpy as np 
from torchvision import models
import torch.nn as nn
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    classify = True
    # Second-stage classifier
    if classify:
        '''
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        '''
        model_ft = models.efficientnet_b0()
        # Get the length of class_names (one output unit for each class)
        output_shape = 16
        # Recreate the classifier layer and seed it to the target device
        model_ft.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=1280, 
                            out_features=output_shape, # same number of output units as our number of classes
                            bias=True)).to(device)

        #num_ftrs = model_ft.fc.in_features
        num_ftrs = 1280
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 16)

        model_ft = model_ft.to(device)

        #model_ft.load_state_dict(torch.load('yolov7_v4_again/classification_comp_best_weights_from_csl.pt', map_location=device)) 
        model_ft.load_state_dict(torch.load('./classification_comp_best_weights_from_csl.pt', map_location=device))        
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(16)]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        for p in pred:
            print('p: ', p)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, model_ft, img, im0s)

        # For activity recognition
        find_activity = True
        if(find_activity):
            pred_with_activity = activity_recognition_run(pred, img, im0s) # last value is the activity
            print('\npred_with_activity : ', pred_with_activity)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {int(c)}{'s' * (n > 1)}, "  # add to string
            
                # ======================================= Omkar - start =======================================
                # create 2D map
                graphics_width = 600
                graphics_height = 1000
                pixel_to_real_scaling = 0.02 # means actual height (meters) = graphics_height * pixel_to_real_scaling
                #pts_transformation_input = np.array([(1492, 372), (2297, 493), (1635, 3472), (100, 997)], dtype = "float32") # in the order (tl, tr, br, bl)
                # MAPPING AGAIN
                #pts_transformation_input = np.array([(1527, 375), (2312, 485), (1716, 3733), (80, 960)], dtype = "float32") # in the order (tl, tr, br, bl)
                #pts_transformation_input = np.array([(1468, 359), (2309, 456), (1579, 2916), (54, 909)], dtype = "float32") # in the order (tl, tr, br, bl)
                pts_transformation_input = np.array([(1614, 392), (2296, 483), (1570, 3165), (88, 983)], dtype = "float32") # in the order (tl, tr, br, bl)
                
                transformation_matrix, warped_im0 = four_point_transform(im0, pts_transformation_input,graphics_height, graphics_width)
                cv2.imwrite('warped.jpg', warped_im0)
                center_coords_all = []
                text_all = []
                print(f'\n\ndet = {det}')
                
                for box_number, (*xyxy, conf, cls) in enumerate(det): 
                    print(xyxy)
                    centroid_x = (xyxy[0] + xyxy[2])/2
                    #centroid_y = (xyxy[1] + xyxy[3])/2
                    #centroid_y =  xyxy[3]

                    if(xyxy[2] < 1367 and xyxy[3] > 1115):
                        centroid_y = (xyxy[1] + xyxy[3])/2
                    else:
                        centroid_y =  xyxy[3]
                    '''
                    if(xyxy[3] > 941):
                        centroid_y = (xyxy[1] + xyxy[3])/2
                    else:
                        centroid_y =  xyxy[3] # consider just the lower horizontal line to be in the plane
                    '''
                    #input_to_transform = (xyxy[])
                    print('Input to transform centroids : ', centroid_x, centroid_y)
                    out_scaled = np.matmul(transformation_matrix, np.array([centroid_x, centroid_y, 1]))
                    out_scaled= out_scaled/out_scaled[2]
                    transformed_point = (int(out_scaled[0]), int(out_scaled[1]))
                    print('Transformed Point : ', transformed_point)
                    center_coords_all.append(transformed_point)

                    text = [f'{[round(pt*pixel_to_real_scaling, 2) for pt in transformed_point]} meters', 
                            f'Cow ID: {int(cls)}' , f'Confidence: {conf:.2f}', f'Activity: {pred_with_activity[box_number]}']
                    text_all.append(text)
                
                print(f'\n\ncenter_coords_all: {center_coords_all}')
                graphics_output = generate_graphics(center_coords_all, text_all, graphics_height, graphics_width)   
                 
                # ======================================= Omkar - end =======================================


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{int(cls)} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
