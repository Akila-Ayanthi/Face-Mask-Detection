import torch
import cv2
import argparse

def parse_opt():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default = None, help="path to image")
    ap.add_argument("--video", type = int, default = 0, help="path to video")
    ap.add_argument("--video_output", default = "webcam_mask_detection.mp4", help="save name of image or video")

    opt = ap.parse_args()
    return vars(opt)

def detect (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. 
            print(f"[INFO] Extracting Bounding Box coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]


            if text_d == 'mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background

                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'nomask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1) ## for text label background

                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

    return frame

    

def main(opt):
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model =  torch.hub.load('ultralytics/yolov3', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection

    classes = model.names

    if opt['image'] != None:
        img_path = opt['image']
        print(f"[INFO] Working with images: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detect(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)

        cv2.namedWindow("Detections on image", cv2.WINDOW_NORMAL) ## creating a free windown to show the result
        # frame = cv2.resize(frame, (416, 416))
        cv2.imshow("Detections on image", frame)

        cv2.waitKey(0)
        print(f"[INFO] Exiting. . . ")
        savename = "det_"+img_path
        cv2.imwrite(savename, frame) ## if you want to save the output result.


    elif opt['video'] !=None:
        vid_path = opt['video']
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


        if opt['video_output']: 
            video_output = opt['video_output']
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = cap.get(cv2.CAP_PROP_FPS)
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
            out = cv2.VideoWriter(video_output, codec, fps, (width, height))

        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret :
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detect(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)
                if video_output:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
                frame_no += 1
        
        print(f"[INFO] Cleaning up. . . ")
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()



if __name__=="__main__":
    opt = parse_opt()
    main(opt)