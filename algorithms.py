import cv2
import numpy as np
import os
import dlib
import urllib.request
import bz2

def harris_corner_detector(image): # Compares gradient of x and y

    # Apply Harris Corner Detector
    harris = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)

    # Threshold to get corners
    threshold = 0.01 * harris.max()
    corners = np.where(harris > threshold)

    # Convert image to color to draw circles
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw circles at corner locations
    for y, x in zip(*corners):
        cv2.circle(image_color, (x, y), 3, (0, 0, 255), -1)

    # Display the result
    return image_color

def shi_tomasi_corner_detector(image): # Compares eigenvalues of structure tensor
    # Apply Shi-Tomasi Corner Detector
    corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.00001, minDistance=10)
    if corners is not None:
        # Convert corners to integers
        corners = corners.astype(int)
    else:
        raise ValueError("No corners were detected.")

    # Convert image to color to draw circles
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw circles at corner locations
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_color, (x, y), 3, (0, 0, 255), -1)

    # Display the result
    return image_color

def fast_corner_detector(image): # Compares intensity of each pixel w surrounding pixels
    # Create FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints
    keypoints = fast.detect(image, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255))

    # Display the result
    return image_with_keypoints

def orb_feature_detector(image): # FAST for keypoint detection, BRIEF for descriptor extraction
    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255), flags=0)

    # Display the result
    return image_with_keypoints

def haar_cascade_face_detector(image): # scan the image at multiple scales and apply a pre-trained model 
# Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjusts how much the image size is reduced at each image scale
        minNeighbors=5,   # Defines how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this are ignored.
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting image with detected faces
    return image

def brute_force_matcher(img1, img2): # Matches keypoints between images using Hamming distance
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create a Brute Force Matcher with Hamming distance and cross-check enabled
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the first 20 matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def flann_matching(img1, img2): # k-d tree algorithm for matching & Lowe’s ratio test to filter good matches.

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Define FLANN based matcher parameters for SIFT descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Perform kNN matching with k=2
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw the good matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

def ransac_outlier_detection(img1, img2):
# Estimates a transformation between two images and identifies inliers (matches consistent with the transformation) versus outliers.
# Combines SIFT keypoint detection, FLANN matching, and RANSAC (Random Sample Consensus)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Define FLANN based matcher parameters for SIFT descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Proceed only if enough matches are found
    if len(good_matches) > 4:
        # Extract location of good matches in both images
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Compute Homography using RANSAC to filter out outliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Draw only the inlier matches (matches that agree with the homography)
        draw_params = dict(matchColor=(0,255,0),   # Inliers in green
                        singlePointColor=None,
                        matchesMask=matchesMask,  # Draw only inliers
                        flags=2)
        img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    else:
        raise ValueError("Not enough good matches found to compute homography.")

    return img_ransac

def dlib_facial_analysis(image):
# Advanced facial analysis using DLib’s pre-trained 68-landmark predictor and face detector


    # Create a directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Define the URL and file paths
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_model_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "models/shape_predictor_68_face_landmarks.dat"

    # Download the compressed model if it doesn't exist
    if not os.path.exists(model_path):
        print("Downloading facial landmark predictor model...")
        urllib.request.urlretrieve(model_url, compressed_model_path)
        
        # Extract the compressed file
        print("Extracting compressed model file...")
        with bz2.BZ2File(compressed_model_path) as input_file, open(model_path, 'wb') as output_file:
            output_file.write(input_file.read())
        
        print(f"Model saved to {model_path}")

      # Helper function to calculate Eye Aspect Ratio (EAR) for blink detection
    def eye_aspect_ratio(eye):
      A = np.linalg.norm(eye[1] - eye[5])  # Distance between vertical landmarks
      B = np.linalg.norm(eye[2] - eye[4])
      C = np.linalg.norm(eye[0] - eye[3])  # Distance between horizontal landmarks
      ear = (A + B) / (2.0 * C)
      return ear

    # Load the predictor and detector
    predictor = dlib.shape_predictor(model_path)
    detector = dlib.get_frontal_face_detector()

    # Convert image to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    # Convert dlib's rectangle to OpenCV style
    def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    # Convert landmark prediction to numpy array
    def shape_to_np(shape):
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        return landmarks

    # Process each face
    for rect in rects:
        # Get face rectangle
        (x, y, w, h) = rect_to_bb(rect)
        
        # Draw face rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get landmarks
        shape = predictor(gray, rect)
        landmarks = shape_to_np(shape)

        # Draw landmarks
        for (lx, ly) in landmarks:
            cv2.circle(image, (lx, ly), 2, (0, 0, 255), -1)

        # Face alignment: Calculate angle using eye centers
        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        cv2.line(image, tuple(left_eye_center), tuple(right_eye_center), (255, 0, 0), 2)
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Blink detection: Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        blink_text = "Blinking" if avg_ear < 0.2 else "Eyes Open"

        # Expression recognition: Check mouth width
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        expression_text = "Smiling" if mouth_width > w * 0.3 else "Neutral"

        # Display analysis results below the face
        text_y = y + h + 20
        cv2.putText(image, f"Rotation: {angle:.1f} degrees", (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 20
        cv2.putText(image, blink_text, (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 20
        cv2.putText(image, expression_text, (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Return the image with analysis displayed
    return image