# Face-Mask-Detection
**Train YoloV3 on Face Mask Dataset**

Clone project and upload Train.ipynb notebook on colab

Run the cells one-by-one

**Deploy YoloV3 for Face Mask Detection**

At the end of training the model weights will be downloaded or you could use the given weights (last.py).

Run command: 
             
             python MaskDetection.py (for mask detection on real-stream videos through webcam)

             python MaskDetection.py --image image_path (for mask detection on images)
             
             python MaskDetection.py --video video_path (for mask detection on videos)
