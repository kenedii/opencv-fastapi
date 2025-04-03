from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
from algorithms import (
    harris_corner_detector,
    shi_tomasi_corner_detector,
    fast_corner_detector,
    orb_feature_detector,
    haar_cascade_face_detector,
    brute_force_matcher,
    flann_matching,
    ransac_outlier_detection,
    dlib_facial_analysis
)

# Initialize the FastAPI app
app = FastAPI()

# Endpoints for single-image algorithms expecting grayscale input
@app.post("/harris_corner_detector")
async def harris_corner_detector_endpoint(file: UploadFile = File(...)):
    """Apply Harris Corner Detector to the uploaded image."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    result = harris_corner_detector(img)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/shi_tomasi_corner_detector")
async def shi_tomasi_corner_detector_endpoint(file: UploadFile = File(...)):
    """Apply Shi-Tomasi Corner Detector to the uploaded image."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    result = shi_tomasi_corner_detector(img)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/fast_corner_detector")
async def fast_corner_detector_endpoint(file: UploadFile = File(...)):
    """Apply FAST Corner Detector to the uploaded image."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    result = fast_corner_detector(img)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/orb_feature_detector")
async def orb_feature_detector_endpoint(file: UploadFile = File(...)):
    """Apply ORB Feature Detector to the uploaded image."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    result = orb_feature_detector(img)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

# Endpoints for single-image algorithms expecting color input
@app.post("/haar_cascade_face_detector")
async def haar_cascade_face_detector_endpoint(file: UploadFile = File(...)):
    """Apply Haar Cascade Face Detector to the uploaded image."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    result = haar_cascade_face_detector(img)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/dlib_facial_analysis")
async def dlib_facial_analysis_endpoint(file: UploadFile = File(...)):
    """Apply Dlib Facial Analysis to the uploaded image."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    result = dlib_facial_analysis(img)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

# Endpoints for two-image algorithms expecting color input
@app.post("/brute_force_matcher")
async def brute_force_matcher_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Apply Brute Force Matcher to the two uploaded images."""
    contents1 = await file1.read()
    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_COLOR)
    contents2 = await file2.read()
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_COLOR)
    result = brute_force_matcher(img1, img2)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/flann_matching")
async def flann_matching_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Apply FLANN Matching to the two uploaded images."""
    contents1 = await file1.read()
    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_COLOR)
    contents2 = await file2.read()
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_COLOR)
    result = flann_matching(img1, img2)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/ransac_outlier_detection")
async def ransac_outlier_detection_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Apply RANSAC Outlier Detection to the two uploaded images."""
    contents1 = await file1.read()
    img1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_COLOR)
    contents2 = await file2.read()
    img2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_COLOR)
    result = ransac_outlier_detection(img1, img2)
    _, encoded_img = cv2.imencode('.jpg', result)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")