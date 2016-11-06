import numpy as np
import cv2
import caffe

# Start the Video Capture
cap = cv2.VideoCapture(2)

# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Caffe Models
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# Mean Setup
mean_filename='mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]
# Age Setup
age_net_pretrained='age_net.caffemodel'
age_net_model_file='deploy_age.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained, mean=mean, channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))
# Gender Setup
gender_net_pretrained='gender_net.caffemodel'
gender_net_model_file='deploy_gender.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained, mean=mean, channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))
# Age and Gender Labels
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

example_image = 'example_image.jpg'
input_image = caffe.io.load_image(example_image)
_ = plt.imshow(input_image)

prediction = age_net.predict([input_image]) 
print 'predicted age:', age_list[prediction[0].argmax()]

prediction = gender_net.predict([input_image]) 
print 'predicted gender:', gender_list[prediction[0].argmax()]

while(True):
    # Read Frame and Write to Display
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
