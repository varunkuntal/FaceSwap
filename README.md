# FaceSwapper
Swap faces between two photos having a face each.

## Get Started
```sh
python main.py -s Images/clinton.jpg -d Images/Trump.jpg -p models/shape_predictor_68_face_landmarks.dat -o ClintonOnTrump.jpg 
```

| Source | Destination | Result |
| --- | --- | --- |
|![](Images/clinton.jpg) | ![](Images/Trump.jpg) | ![](Output/ClintonOnTrump.jpg) |

| Source | Destination | Result |
| --- | --- | --- |
|![](Images/o.jpg) | ![](Images/dt.jpg) | ![](Output/ObamaOnTrump.jpg) |

| Source | Destination | Result |
| --- | --- | --- |
|![](Images/db2.jpg) | ![](Images/s.jpg) | ![](Output/beckamOnSuperman.jpg) |

| Source | Destination | Result |
| --- | --- | --- |
|![](Images/db2.jpg) | ![](Images/lm1.jpg) | ![](Output/BeckhamOnMessi.jpg) |

| Source | Destination | Result |
| --- | --- | --- |
|![](Images/rg1.jpg) | ![](Images/v3.jpg) | ![](Output/RagOnVoldy.jpg) |

## Install
### Requirements
* [dlib](http://dlib.net/)
* OpenCV 3
* Numpy

Note: See [requirements.txt](requirements.txt) for more details.

