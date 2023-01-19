import cv2 
import numpy as np
from itertools import chain
import sys
import serial
import time

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            print("Resizing image to 28x28 scale...")
            img_ = cv2.resize(gray,(28,28))
            print("Resized...")
            img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")
        
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
img = cv2.imread('imageA.jpg', 2)
imglist=img.tolist()

imgtosend=np.array(img)

#print((250).to_bytes(1, 'little'))
# for i in range(length):
#       array= hex_array[i].to_bytes(1, 'little')




############################### PYserial
SerialObj = serial.Serial("/dev/ttyACM0")
print(SerialObj) #display default parameters
time.sleep(3) 
SerialObj.baudrate = 115200  # set Baud rate to 9600
SerialObj.bytesize = 8     # Number of data bits = 8
SerialObj.parity   ='N'    # No parity
SerialObj.stopbits = 1     # Number of Stop bits = 1

print(SerialObj)
SerialObj.flush()
SerialObj.flushInput()
SerialObj.flushOutput()

img=img.flatten().astype(np.uint8)
SerialObj.write(img.tobytes())
# time.sleep(1)

# while(1):
#     myBytes = SerialObj.read(5)
#     print(myBytes)


# hex_array = [hex(x) for x in img]   #list of hexadecimal strings
# length=len(hex_array)
# array=[]
# array = [0 for i in range(length)] 

# for i in range(length):
#     array[i]=(hex_array[i].encode())
#     SerialObj.write(hex_array[i].encode())

# array = np.array([1,2,3], dtype=np.int8)
# print(array.tobytes())
# SerialObj.write(array.tobytes())


# time.sleep(1)
# while(1):
#     myBytes = SerialObj.read(20)
#     print(myBytes)



# while True:
#     #SerialObj.flush()
#     #Reads one byte of information
#     SerialObj.write('h'.encode())
#     time.sleep(1)
#     # for i in range(length):
#     #     array[i]=(hex_array[i].encode())
#     #     SerialObj.write(hex_array[i].encode())

#     myBytes = SerialObj.read(1)
#     print(myBytes)
    # Checks for more bytes in the input buffer
    #bufferBytes = SerialObj.inWaiting()
# If exists, it is added to the myBytes variable with previously read information
    # if bufferBytes:
    #     myBytes = myBytes + ser.read(bufferBytes)
    #     print(myBytes)
    # for i in range(length):
    #     array[i]=(hex_array[i].encode())
    #     SerialObj.write(hex_array[i].encode())
        
    #print(array)
    #data_raw = SerialObj.read(15)
    #print(data_raw)

