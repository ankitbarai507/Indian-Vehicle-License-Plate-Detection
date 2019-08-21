# Indian-Vehicle-License-Plate-Detection
Challenge done as part of TCS HumAIn challenge

Download tesseract OCR for windows from [here](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0.20190526.exe)
It makes use of python-Tesseract OCR. \\
For that change the path of Tesseract OCR in files license_detection.py and yolo_license_detection.py to where it installed on your system.

Weights files for the model can be accesed [here](https://drive.google.com/drive/folders/11Y3Dmp4BPTZzpo4TLB328OESpx9k0dkJ?usp=sharing)

Put yolo-v3-tiny_last.weights files inside yolo-coco/   \\
Put other two files directly in the main folder.

To detect license plate having a single car use mainfile.py \\
python mainfile.py --image images/car.png

To detect license plate  for each vehicle in a image having multiple cars use multiple_cars.py
