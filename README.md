# CelebrityDoppelganger

To identify the celebrity doppelgangers of Sofia Solares and Shashikant Pedwal, I created two functions. The first function is to create a vector that describes the features of the face (encode_faces). The second function is to calculate the cosine similiarity between the faces of interest with the other celebrity faces. 

For the encode_face function, I imported all of the celebrity faces. I then converted the images from RGB2BGR as dlib uses BGR by default. Before encoding the features of a face, a face must be identified. To locate a face in a photograph one must find the coordinate of the face in the image. This can be done by placing a bounding box around the face. For any given photograph,  a face detection system will output zero or more bounding boxes that contain faces. Detected faces can then be provided as input to a subsequent system, such as a face recognition system. Next, we can find the face encoding values from the faces in the images. This approach will return 128 values per face from images. This object maps human faces into 128D vectors, so that images of the same person have similar embeddings to each other and images of different people are mapped far apart.

```python
# Compute the 128-dimension face encoding for each faces
def encode_faces(path_):
    # Known face dataset
    faces_encodings = {}
    # All the folders inside the path
    allCelebs = [d for d in os.listdir(path_)
                    if os.path.isdir(os.path.join(path_, d))]
    # For each folder
    for i, celebID in enumerate(allCelebs):
        # Get all the images in the folder
        currPath = os.path.join(path_, celebID)
        allImages = glob.glob(os.path.join(currPath, '*.JPEG'))
        #For each images
        for pathImg in allImages:
            # Load the image into a numpy array
            image = dlib.load_rgb_image(pathImg)
            # Compute the 128-dimension face encoding for each faces
            image_bgr= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = faceDetector(image_bgr, 1)
            if len(dets) == 1:
              # We keep only the images with exactly 1 face
              filename = pathImg.replace('\\', '/').split('/')[-1]
              celeb_ID = filename.split('_')[0]
              
              # Now process each face we found.
              for k, d in enumerate(dets):
                # Get the landmarks/parts for the face in box d.
                shape = shapePredictor(image_bgr, d)
          
                # Compute the 128D vector that describes the face in img identified by shape.  
                face_descriptor = faceRecognizer.compute_face_descriptor(image_bgr, shape)
                faces_encodings[(filename, labelMap[celeb_ID])] = face_descriptor
    return faces_encodings

```
Once we have calculated the 128D vector embedding for each image, the next function created one 128D vector embedding for the celebrities of interest. To identify the celebrity doppelganger, I calculated the cosine similiarity between each test image and the remaining image. The two images with the maximum cosine similiarity returned the image of the celebrity doppelganger. Why cosine similiarity? Cosine similarity is a metric used to measure how similar the vectors are irrespective of their size. It represents the cosine of the angle between two vectors projected in multi-dimensional space. The cosine similarity is advantageous because even if the two similar vectors are far apart by the Euclidean distance, chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.


```python



from numpy import dot
from numpy.linalg import norm
# read image
testImages = glob.glob('test-images/*.jpg')

test_descriptors = []
test_face_array=[]

for test in testImages:
    print(test)
    im = cv2.imread(test)
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    #####################
    #  YOUR CODE HERE
    dets = faceDetector(imDlib, 1)
    # Now process each face we found.
    for k, d in enumerate(dets):
      # Get the landmarks/parts for the face in box d.
      shape = shapePredictor(imDlib, d)

      # Compute the 128D vector that describes the face in img identified by shape.  
      test_face_descriptor = faceRecognizer.compute_face_descriptor(imDlib, shape)
      #faces_encodings[(filename, labelMap[celeb_ID])] = test_face_descriptor
      descriptor_ratio=[]
      for cds in celeb_face_encoding.values():
        #calculate Cosine Similarity
        cos_sim = dot(cds, test_face_descriptor)/(norm(cds)*norm(test_face_descriptor))
        descriptor_ratio.append(cos_sim)
    
    print("max cosine similiarity",max(descriptor_ratio))
```


With this approach, Sofia Solares had the highest cosine similiarity (0.94) with Selena Gomez. Shashikant Pedwal had the highest cosine similiarity (0.93) with Amitabh Bachchan. 
