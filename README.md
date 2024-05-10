### SC-YOLO

We use the python=3.8.17,pytorch =2.0.1 and cuda=11.8.

If you want to use SC-YOLO, please pip ultralytics.

After pip is finished, please replace the ultralytics  you just installed with this one.

The SC-YOLO network is shown as yolov8s-SC-YOLO.yaml.

If you want to see the  code for SC-YOLO's components, you can open the python file under ultralytics/nn/modules.

For ultralytics dataset training format you can refer to this URL https://docs.ultralytics.com/.

You can use command in the terminal:

yolo classify train data=yourdataset model=yolov8s-SC-YOLO.yaml epochs=XXXX batch=XXXXX 
