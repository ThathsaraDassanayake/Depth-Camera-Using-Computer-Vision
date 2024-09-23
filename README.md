# Depth-Camera Using Computer Vision

![image](https://github.com/user-attachments/assets/1bb8ec3f-4448-41b2-be54-42bb0ab67134)
The project focuses on designing a Depth Camera using computer vision techniques, object detection, and feature matching to create a portable and cost-effective solution for accurate depth measurement. The key aspects of the project include:

1.  Hardware Design:

* The project involved selecting and configuring components such as a stepper motor, microcontroller (ATmega328P), and a USB Type-C adapter.
* A 3D-printed cylindrical enclosure was designed to house the components, ensuring structural integrity and ease of assembly.

2. Software Integration:
* Algorithms were developed for image capturing, camera calibration, object detection using YOLOv8, feature detection using SIFT algorithm, feature matchig using BFMatcher with NORM L2 norm., and depth calculation.
* The camera calibration process involved both mono and stereo calibration, providing the necessary parameters for accurate depth measurement.

3. System Optimization:

* The project included power management strategies and optimization of overall system performance.
* Integration of the stepper motor allowed the camera to capture images from multiple angles, aiding in depth calculation.

4. Testing and Evaluation:
* Extensive testing was conducted to ensure the accuracy and robustness of the system, including calibration, object detection, feature matching, and motor control.
* The YOLOv8 model was custom-trained to improve object detection accuracy, a critical component of the system.

5. Group Collaboration:
* The project was a team effort where each member contributed their expertise to different aspects of the design, implementation, and testing.
* Team members collaborated on problem-solving, particularly in areas like camera calibration, motor integration, and algorithm optimization.

https://github.com/user-attachments/assets/b677afcc-1f56-4e24-a2f9-319fe8ce68ad

## Software Integration and Programming Analysis
This section provides a comprehensive overview of the programming aspects of our depth camera
project, detailing the implementation and testing processes involved in capturing images, calibrating cameras, detecting objects, matching features, and calculating depth.

1. Capturing the Image
* Implement a code to capture a photo from the USB webcam when the ’S’ key is pressed.
* Used OpenCV to interact with the USB webcam and capture images. The
code allows capturing images by pressing the ’S’ key and exits on pressing ’Q’ or interrupt.

[Image Capture Code](https://github.com/Thathsara-Dassanayake/Depth-Camera/blob/main/FinalCode.py)

2. Camera Calibration
* Calibrated the camera to obtain intrinsic and extrinsic parameters.
* Mono Camera Calibration
   * Used a chessboard pattern for calibration.
   * Detected chessboard corners in multiple images.
   * Computed the camera matrix and distortion coefficients
* Stereo Camera Calibration
   * Calibrated two cameras simultaneously to obtain the rotation and translation matrices
between them.

[Camera Calibration Code](https://github.com/Thathsara-Dassanayake/Depth-Camera/blob/main/calibration.py)

* Results
  * The calibration process provided the camera matrix and distortion coefficients for each
camera.
  * The stereo calibration provided the rotation and translation matrices between the two
cameras.

![image](https://github.com/user-attachments/assets/60ea3dfe-c4a8-4d10-8e19-76003b379a06)


3. Object Detection with YOLOv8
* Implemented object detection using the YOLOv8 model, custom-trained to detect
specific objects.

* Model Training
  * Used a custom dataset created using RoboFlow.
  * Trained the YOLOv8 model on this dataset, adjusting parameters like learning rate
and batch size for optimal performance.

* Object Detection
  * Implemented the object detection code to output bounding boxes

[Object Detection Code](https://github.com/Thathsara-Dassanayake/Depth-Camera/blob/main/ObjectDetection.py)

* Results
![image](https://github.com/user-attachments/assets/141a417b-f740-4c2b-8d59-7cafb8c381ea)

4. Feature Matching
* Feature Detection Algorithms
  * Experimented with SIFT, ORB, and SURF for feature detection
  * Chose SIFT for its robustness and accuracy.
* Feature Matching
  * Implemented feature matching using the BFMatcher with NORM L2 norm.

[Feature Matching Code](https://github.com/Thathsara-Dassanayake/Depth-Camera/blob/main/FinalCode.py)

* Results
![image](https://github.com/user-attachments/assets/7c4390e2-d209-43a7-922f-3ac9e37a05cb)

5. Depth Calculation
* Calculated the depth of points using triangulation based on the matched features
and camera parameters.
* It was done using triangulation
  * Used the camera matrix, rotation matrix, and translation matrix to triangulate
matched features and calculate depth.

[Depth Calculation Code](https://github.com/Thathsara-Dassanayake/Depth-Camera/blob/main/FinalCode.py)

6. Stepper Motor Integration Using C++
* Rotate the camera to two positions using a stepper motor controlled by an ATmega
microcontroller.
* Motor Control
  * Implemented motor control to rotate the camera.
  * Used an ATmega microcontroller to control the stepper motor.

[StepperMotor](https://github.com/Thathsara-Dassanayake/Depth-Camera/blob/main/StepperMotor.cpp)

* Results
  * Successfully rotated the camera to capture two images from different positions.

## Circuit Analysis
1. Power supply
* Our power supply is a fairly simple setup. As the input, we use a 12v power supply (provided through
a power-regulated power supply) and then inside the device by using a L7812 voltage regulator we
also make our own 5v power supply inside the circuit. Then we use this to power up the atmega328
chip and servo motor driver. The primary components include a voltage regulator, capacitors for
filtering, and connectors for input and output.

![image](https://github.com/user-attachments/assets/f2fd59a6-6eae-4b8c-913c-2ede747f27ba)

2. Stepper motor driver carrier
The stepper motor driver drives the stepper motor, converting digital signals from the microcontroller
into precise movements. It features the A4988 stepper motor driver, which controls the current flow
to the motor windings, enabling accurate positioning

![image](https://github.com/user-attachments/assets/35ee1f55-68dd-4772-9ae7-aa44e070a040)

3. Microcontroller Unit
This module incorporates the ATmega328P microcontroller, which serves as the central processing
unit for the circuit. It receives power from the power supply, controls the motor driver, and interfaces
with other peripherals. The tactile switch SW2 could be used to start or stop the motor, change
modes of operation, or initiate specific functions in the program running on the microcontroller.
Here, we use it as a reset button. Other key components include the ATmega328P, crystal oscillator
for clock generation, and decoupling capacitors.

![image](https://github.com/user-attachments/assets/9aaafe7c-4b75-4871-a910-b71f084a164d)

4. Type C adapter
The Type C adapter circuit facilitates communication between the microcontroller and external
devices through a USB Type C connector. It includes an RS232 transceiver for serial communication,
capacitors for stabilization, and connectors for the USB interface. In this adapter circuit, the CH340
serves as an interface for serial communication. It converts USB signals from a host device (such as
a computer) into UART signals that the microcontroller can understand and vice versa.

![image](https://github.com/user-attachments/assets/ebc0d170-6d36-4aaa-8fa0-059b34e9ead0)

## PCB Design
Using the above schematics, we designed a PCB that will fit to be implemented as a rotating sensor.
We carefully chose the dimensions and shape for the PCB. We also carefully planned the power
supply method for the PCB. Since the stepper motor driver needs a 12V power supply, we designed
the power module and the paths to be suitable to carry the needed voltage. The enclosure will have
a barrel jack, and from that, the power will be supplied to the designated headers on the PCB.
We also implemented the USB-C port and its connection such that we can control the rotation of
the stepper motor from our PC. We placed mounting holes on the PCB to support the structure
during movements.

![image](https://github.com/user-attachments/assets/8d545e9a-7093-46bf-b61f-f79814c0b21a)
![image](https://github.com/user-attachments/assets/e1cfb91b-1e08-441f-bde2-78ed69c5f46d)
![image](https://github.com/user-attachments/assets/a57a3946-7a24-4180-80c0-fcbe2012a984)


## Enclosure analysis

All design of the enclosure was done by SolidWorks.

![image](https://github.com/user-attachments/assets/8e5ed0e0-11df-495b-9fad-dbd27774d084)
![image](https://github.com/user-attachments/assets/5e643612-cb25-4ab9-8224-493434081466)
![image](https://github.com/user-attachments/assets/98460c83-db43-494a-8e55-e40fb1b08aba)
![image](https://github.com/user-attachments/assets/ef7d76ff-9db4-452e-9b2b-7ebe0c750134)
![image](https://github.com/user-attachments/assets/7d3f7956-c9f8-43da-80a7-1d40c64d0561)
![image](https://github.com/user-attachments/assets/e0789a84-88ad-4fcc-9d43-5ca6dfa51247)

## Complete Integrated Design
The complete 3D printed design integrates the main body, rotating component, and back lid seamlessly. The assembly clearly shows the PCB and stepper motor, highlighting their well-considered
placement and integration within the enclosure. This comprehensive design ensures that all components work together cohesively, providing a robust and efficient solution for the depth camera
system.

![image](https://github.com/user-attachments/assets/be27f6ee-c382-4ef0-aa73-9c1ff617cca9)
