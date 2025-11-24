##### Fashion Coordinator AI

## Explanation 
Fashion recommendation system using YOLOv8 and OpenCV.
This AI detects the color of your top and bottom clothing and evaluates the color harmony based on color theory.

## Key Features
- Real-time Person Detection: Uses YOLOv8 to detect humans in the frame.
- Smart Area Splitting: Automatically separates Top (Upper body) and Bottom (Lower body) areas based on body height.
- Advanced Color Analysis: Uses 5x5 Grid Voting to accurately detect colors.
- Fashion Stylist: Evaluates color combinations (Tone-on-Tone, Earth Look, Contrast, etc.) and provides feedback.


## Points of Attention/Limit
Depending on the amount of light in the room, the results may vary. In this case, you can adjust the camera illumination by writing an integer between -1 and -13 in ```cap.set (cv2.CAP_PROP_EXPOSURE, -6.0)``` part of the code.

## Environment
- OS: Windows 11
- Language: Python 3.13.9
- IDE: VS Code
- Key Libraries:
    + ultralytics (YOLOv8)
    + opencv-python (Image Processing)
    + numpy (Calculation)

## Installation
1. Clone the repository
   ```git clone https://github.com/milh0919-stack/My_project_folder.git```
   ```cd My_project_folder```


2. Setup Conda Environment
    ```conda create -n fashion_ai python=3.10 -y```
    ```conda activate fashion_ai```


3. Install Dependencies
    ```pip install -r requirements.txt```

## Usage
Type ```python maincode.py``` in the terminal
To quit, press 'q' botton
