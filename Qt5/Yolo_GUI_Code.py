import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage 
import cv2
import numpy as np
from ultralytics import YOLO
import os 

class YoloApp(QDialog):
    def __init__(self):
        super(YoloApp, self).__init__()
        uic.loadUi(r'E:\Diploma\Machine Learning\AMIT Final Project\Qt5\Yolo_GUI_App.ui', self)  # Load your .ui file here

        # Load buttons
        self.load_button_1 = self.findChild(QPushButton, 'pushButton')  # Button for loading first image
        self.load_button_1.clicked.connect(self.load_image)

        self.load_button_2 = self.findChild(QPushButton, 'pushButton_2')  # Button for model predict
        self.load_button_2.clicked.connect(self.predict_with_model)

        self.match_button = self.findChild(QPushButton, 'pushButton_3')  # Button for VGG16 predict
        #self.match_button.clicked.connect(self.predict_with_vgg)

        # Image viewer
        self.image_viewer_1 = self.findChild(QGraphicsView, 'graphicsView')  # QGraphicsView for image
        self.scene = QGraphicsScene(self)
        self.image_viewer_1.setScene(self.scene)

        # Load models
        self.model_path = r'E:\Diploma\Machine Learning\AMIT Final Project\Dataset\Results6_V8s\weights\best.pt'
        self.output_path= r'E:\Diploma\Machine Learning\AMIT Final Project\Qt5\Output'

        # Class names mapping
        self.classes = {0: 'Wearing Mask', 1: 'Not Wearing Mask', 2: 'Wearing Mask Incorrectly'}

        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if self.image_path:
            # Load and resize image for viewing
            img = cv2.imread(self.image_path)
            img_resized = cv2.resize(img, (448, 448))  # Resize to 448x448 for display
            height, width, _ = img_resized.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(q_img))
            self.image_viewer_1.setScene(self.scene)

    def predict_with_model(self):
        if self.image_path:
            # Initialize the YOLO model
            model = YOLO(self.model_path)

            # Make predictions on the image
            predictions = model.predict(source=self.image_path, conf=0.6, save=True,save_dir=self.output_path)
            
            
            # Process predictions and draw bounding boxes
        if predictions:  # Check if predictions exist
            img = cv2.imread(self.image_path)  # Read the original image
            boxes = predictions[0].boxes  # Get bounding boxes from the prediction
            if len(boxes) > 0:  # If any boxes are detected
                result_text = ""  # Initialize an empty string to store all results

                # Define a color mapping for the classes
                class_colors = {
                    0: (255, 0, 0),  # Class 0: Blue
                    1: (0, 255, 0),  # Class 1: Green
                    2: (0, 0, 255),  # Class 2: Red
                }

                for box in boxes:
                    # Convert box.xyxy from tensor to numpy array, then to integers
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  
                    class_idx = int(box.cls)  # Get the class index
                    confidence = float(box.conf) * 100  # Get confidence score

                    # Get the color for the current class
                    color = class_colors.get(class_idx, (255, 255, 255))  # Default to white if class is undefined

                    # Draw the bounding box on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Convert color to hex for HTML
                    color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"  # Convert BGR to RGB hex

                    # Add label and confidence score text with HTML color and custom font size
                    font_size = 2  # Set the font size
                    label_text = f"<font color='{color_hex}' size='{font_size}'>{self.classes[class_idx]}: {confidence:.2f}%</font>"
                    result_text += f"{label_text}<br>"  # Append the text for each detection

                # Update the label once after the loop
                self.result_label.setText(f"<html>{result_text}</html>")
                self.result_label.setWordWrap(True)

                # Show the result image with bounding boxes in the GUI
                img_resized = cv2.resize(img, (448, 448))  # Resize for displaying in GUI
                height, width, _ = img_resized.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                self.scene.clear()
                self.scene.addPixmap(QPixmap.fromImage(q_img))
                self.image_viewer_1.setScene(self.scene)

            else:
                # If no objects detected, display a blank result or message
                self.result_label.setText('No objects detected')
        else:
            self.result_label.setText('Error in prediction')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloApp()
    window.show()
    sys.exit(app.exec_())
