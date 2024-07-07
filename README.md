# Steps to Run Code

## Clone the Repository

To clone the repository, run the following command in your terminal:

```bash
git clone https://github.com/asavaribhelawe/Person-detection
```
## Go to the Cloned Folder
Navigate into the cloned repository folder:
```bash
cd Person-detection
```
## Install Requirements
Install the required Python packages by running:
```bash
pip install -r requirements.txt
```
## Download the Pre-trained YOLOv9 Model Weights
Download the YOLOv9 weights from the following link and place them in your project directory:
[Download YOLOv9 weights](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)

## Downloading the DeepSORT Files
Download the DeepSORT files by running the following command:
```bash
gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
```
After downloading the DeepSORT Zip file from the drive, unzip it by running the script.py file in yolov9 folder.

## Running the Code
use the following command:

```bash
python detect_dual.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device cpu
```
## Object Counting

Output files will be created in the working-dir/runs/detect/obj-tracking directory with the original filename.
