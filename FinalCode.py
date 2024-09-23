import cv2 as cv 
import time
import numpy as np
import serial.tools.list_ports

from ultralytics import YOLO

def ImageCapture(input_string):
    key = cv. waitKey(1)
    webcam = cv.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            #print(check) #prints true as long as the webcam is running
            #print(frame) #prints matrix values of each framecd 
            cv.imshow("Capturing", frame)
            key = cv.waitKey(1)
            if key == ord('s'): 
                cv.imwrite(filename= input_string, img=frame)
                webcam.release()
                cv.waitKey(1650)
                cv.destroyAllWindows()
            
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv.destroyAllWindows()
            break

ImageCapture('left1.jpg')

# Function to list available serial ports
def list_ports():
    ports = serial.tools.list_ports.comports()
    ports_list = []
    for one_port in ports:
        ports_list.append(str(one_port))
        print(str(one_port))
    return ports_list

# Function to select a port
def select_port(ports_list):
    val = input("Select Port: COM")
    for x in range(0, len(ports_list)):
        if ports_list[x].startswith("COM" + str(val)):
            port_var = "COM" + str(val)
            print(port_var)
            return port_var
    return None

# Function to send a command to the Arduino
def send_command(serial_instance, command):
    serial_instance.write(command.encode('utf-8'))

# Main script
ports_list = list_ports()
port_var = select_port(ports_list)

if port_var:
    serial_inst = serial.Serial()
    serial_inst.baudrate = 9600
    serial_inst.port = port_var
    serial_inst.open()

    # Set pin to high state
    send_command(serial_inst, 'H')  # Assuming 'H' sets the pin high
    print("Pin set to HIGH state.")
    time.sleep(1)  # Wait for 1 second or the duration you need

    serial_inst.close()
else:
    print("No valid port selected.")

time.sleep(10) #waiting time to let the camera rotate to the other position

ImageCapture('right2.jpg')     #taking the right picture once the camera is done rotating

# Predict using the trained model
trained_model_path = "/Users/sasindu/Desktop/DC/Results/detect/train/weights/best.pt"  
trained_model = YOLO(trained_model_path)
results = trained_model.predict(["/Users/sasindu/Desktop/DC/left.jpg","/Users/sasindu/Desktop/DC/right.jpg"],save=True, imgsz=640, conf=0.98)


# Extract bounding boxes, classes, names, and confidences
boxes1 = results[0].boxes.xywh.tolist()
boxes2 = results[1].boxes.xywh.tolist()


def center_to_topleft(cx, cy, w, h):
    
    x = cx - w / 2
    y = cy - h / 2
    bbx1 = [int(x),int(y),int(w),int(h)]
    return bbx1

#Bounding Boxes

bbox1= center_to_topleft(boxes1[0][0], boxes1[0][1], boxes1[0][2], boxes1[0][3]) 
bbox2= center_to_topleft(boxes2[0][0], boxes2[0][1], boxes2[0][2], boxes2[0][3]) 

def match_features_in_bboxes(img1, img2, bbox1, bbox2, distance_threshold=0.4):
    """
    Match SIFT features within specified bounding boxes in two images, filter by match quality,
    and prepare for triangulation using the BFMatcher with NORM_L2.
    
    Args:
    img1, img2: Input images.
    bbox1, bbox2: Bounding boxes in each image as (x, y, width, height).
    distance_threshold: Ratio threshold for filtering good matches based on Lowe's ratio test.
    
    Returns:
    pts1, pts2: Arrays of the matched keypoints from each image, ready for triangulation.
    """
    # Extract regions of interest
    roi1 = img1[bbox1[1]:bbox1[1] + bbox1[3], bbox1[0]:bbox1[0] + bbox1[2]]
    roi2 = img2[bbox2[1]:bbox2[1] + bbox2[3], bbox2[0]:bbox2[0] + bbox2[2]]
    
    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect and compute SIFT features in ROIs
    kp1, des1 = sift.detectAndCompute(roi1, None)
    kp2, des2 = sift.detectAndCompute(roi2, None)

    # Create BFMatcher object with distance norm L2
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    # Match descriptors and apply Lowe's ratio test
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < distance_threshold * n.distance:
            good_matches.append(m)

    # Extract coordinates of matched keypoints
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Adjust points to the full image coordinates
    pts1 += np.array([bbox1[0], bbox1[1]])
    pts2 += np.array([bbox2[0], bbox2[1]])

    # Draw matches
    match_img = cv.drawMatches(roi1, kp1, roi2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the match image
    match_img_path = "/Users/sasindu/Desktop/DC/match_features.jpg"
    cv.imwrite(match_img_path, match_img)

    return pts1, pts2

img1 = cv.imread("/Users/sasindu/Desktop/DC/left.jpg", cv.IMREAD_GRAYSCALE)  # Left image
img2 = cv.imread("/Users/sasindu/Desktop/DC/right.jpg", cv.IMREAD_GRAYSCALE)  # Right image

pts1, pts2 = match_features_in_bboxes(img1, img2, bbox1, bbox2)

R = np.array([[ 1, 0, 0],
 [ 0,  1, 0],
 [0,  0,  1]])

T = np.array([[-60],
 [ 0],
 [ 0]])

mtx = np.array([[4.06850113e+03, 0.00000000e+00, 2.11799463e+03],
 [0.00000000e+00, 4.07285720e+03, 2.84485951e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

P1 = np.hstack((mtx, np.zeros((3, 1))))  # Projection matrix for the first camera
P2 = mtx @ np.hstack((R, T.reshape(-1, 1)))  # Projection matrix for the second Position

pts1 = pts1.T
pts2 = pts2.T

# Triangulate points
points4D = cv.triangulatePoints(P1, P2, pts1, pts2)
points3D = points4D[:3] / points4D[3]
depths = points3D[2, :]  # Extract Z (depth information)

#Printing Depth
print("Depth :", np.mean(depths))
