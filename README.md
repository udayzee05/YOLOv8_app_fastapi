# Real-time People Counter using YOLOv8 and FastAPI

This repository contains code for a real-time object detection application that counts people using the YOLOv8 algorithm and the FastAPI framework. The output can be accessed through a web browser, making it easy to use and accessible from anywhere.

The app can be easily extended to other use cases beyond people counting by training the model on the desired object. The app is designed to be scalable, so it can handle large volumes of input data and run efficiently on different devices.

## Getting Started

To run this app, you will need to have Python installed on your system, along with the following packages:

- FastAPI
- uvicorn
- opencv-python
- numpy
- ultralytics

Once you have installed these packages, you can run the app by executing the following command in the terminal:

```bash
python main.py
```
This will start the app and you can access the output by visiting http://localhost:8000 in your web browser.

### Usage
To use the app, simply upload a video or live stream and the app will count the number of people in real-time. The app also provides an option to download the video with the bounding boxes drawn around the detected objects.

You can ulso use this app with rtsp url just pass rtsp url to videoCapture ()

### Extending the App
This app can be easily extended to detect and count other objects by training the YOLOv8 model on the desired object. You can find more information on how to do this in the YOLOv8 repository.

### Contributing
Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
