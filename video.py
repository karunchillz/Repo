import numpy as np
import cv2
import time
import requests
import operator
from threading import Timer

_url = 'https://api.projectoxford.ai/vision/v1/analyses'
_key = 'c360270a2fcd4797ac7f16dbdac7c50d'
_maxNumRetries = 10
menAds = ['men-1.jpg','men-2.jpg','men-3.jpg','men-4.jpg','men-5.jpg']
womenAds = ['women-1.png','women-2.jpg','women-3.JPG','women-4.png','women-5.jpg']
neutralAds = ['Neutral-1.jpg','Neutral-2.jpg', 'Neutral-3.jpg','Neutral-4.jpg','Neutral-5.jpg','Neutral-6.jpg','Neutral-7.jpg','Neutral-8.jpg','Neutral-9.jpg','Neutral-10.jpg']

menIndex = 0
womenIndex = 0
neutralIndex = 0

# Display images within Jupyter
def processRequest( json, data, headers, params ):

    retries = 0
    result = None
    while True:
        response = requests.request( 'post', _url, json = json, data = data, headers = headers, params = params )
        if response.status_code == 429: 
            print( "Message: %s" % ( response.json()['error']['message'] ) )
            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break
        elif response.status_code == 200 or response.status_code == 201:
            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            #print( "Message: %s" % ( response.json()['error']['message'] ) )
        break
    return result

def renderResultOnImage( result):
    if 'description' in result:
	print result['description']['captions']
    if 'faces' in result:
	maleNumber = 0
	femaleNumber = 0
	for face in result['faces']:
	    print face['gender']
	    print face['age']
	    if face['gender'] == 'Male':
		maleNumber = maleNumber + 1
	    else:
	        femaleNumber = femaleNumber + 1
	if femaleNumber > maleNumber:	
    	    cv2.namedWindow("Channels")
    	    cv2.moveWindow("Channels",600,300)
	    img123 = cv2.imread(womenAds[womenIndex],0)
	    womenIndex = womenIndex + 1
    	    cv2.imshow("Channels", img123 )
	else:
    	    cv2.namedWindow("Channels")
    	    cv2.moveWindow("Channels",600,300)
	    img123 = cv2.imread(menAds[menIndex],0)
	    menIndex = menIndex + 1
    	    cv2.imshow("Channels", img123 )	
    else:			
        cv2.namedWindow("Channels")
        cv2.moveWindow("Channels",600,300)
	img123 = cv2.imread(neutralAds[neutralIndex],0)
	neutralIndex = neutralIndex + 1
        cv2.imshow("Channels", img123 )	
	
def callVision():
    # Load raw image file into memory
    pathToFileInDisk = '/home/root/Desktop/code/color_image.png'
    with open( pathToFileInDisk, 'rb' ) as f:
        data = f.read()
    
    # Computer Vision parameters
    params = { 'visualFeatures' : 'Faces,Categories'} 

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key
    headers['Content-Type'] = 'application/octet-stream'
    json = None

    result = processRequest( json, data, headers, params )
    if result is not None:
        renderResultOnImage( result)
    else:			
        cv2.namedWindow("Channels")
        cv2.moveWindow("Channels",600,300)
	img123 = cv2.imread(neutralAds[neutralIndex],0)
	neutralIndex = neutralIndex + 1
        cv2.imshow("Channels", img123 )	
    Timer(10.0, callVision).start()     


# Start the Video Capture
cap = cv2.VideoCapture(2)

# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
smilePath = "haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
smileCascade = cv2.CascadeClassifier(smilePath)

Timer(10.0, callVision).start()

while(True):
    # Read Frame and Write to Display
    ret,frame = cap.read()
    cv2.namedWindow("Main")
    cv2.moveWindow("Main",10,10)
    cv2.imshow("Main",frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags= cv2.CASCADE_SCALE_IMAGE
            )

        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            print "Found", len(smile), "smiles!"
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Main", frame)
    cv2.imwrite('color_image.png',frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
