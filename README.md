# opencv-api

FastAPI endpoints for running these CV algorithms on a photo:
- harris_corner_detector
- shi_tomasi_corner_detector
- fast_corner_detector
- orb_feature_detector
- haar_cascade_face_detector
- brute_force_matcher
- flann_matching
- ransac_outlier_detection
- dlib_facial_analysis


Dependencies:
- Python (3.11.5 used)
- Pip (23.2.1 used)

How to run: 

Windows:
- Double click launch_api.bat

Other:
- pip install -r requirements.txt
- uvicorn api:app --port=8888
