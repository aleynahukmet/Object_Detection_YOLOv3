# Object_Detection_YOLOv3

Other methods for object detection, like R-CNN and its variations, used a pipeline to perform this task in multiple steps. This can be slow to run and also hard to optimize, because each individual component must be trained separately.
And in case of YOLO the name itself explains a lot about it, that it just goes through the entire image just once. In this project I implemented YOLO v3 for object detection.

# Requirements

-Python
-OpenCV

YOLO trained on the COCO dataset. The COCO dataset consists of 80 labels. Beside coco.names file we will also need weight and cfg files.

# Demo Output

<p align="center"> 
   <img alt="Ekran Resmi 2021-06-28 01 15 28" src="https://user-images.githubusercontent.com/87663976/160645978-9040621e-f082-40bf-8a42-b92e41b99bf5.png">
</p>
