# email_from = "secalertai@gmail.com"
# email_to = "secalertai@gmail.com"
# email_password = "oohifmdsxyuymshj"


import cv2
import numpy as np
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Email configuration
smtp_port = 587
smtp_server = "smtp.gmail.com" 
email_from = "secalertai@gmail.com"
email_to = "secalertai@gmail.com"
email_password = "oohifmdsxyuymshj"

# Fire detection configuration
fire_reported = False
fire_detected = False
email_sent = False

simple_email_context = ssl.create_default_context()

def send_email(image_path):
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=simple_email_context)
            server.login(email_from, email_password)

            # Open the image file and attach it to the email
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data, name="fire.jpg")

            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = email_from
            msg['To'] = email_to
            msg['Subject'] = "Fire Detected"
            msg.attach(image)

            # Send the email
            server.sendmail(email_from, email_to, msg.as_string())
        
        print(f"Email successfully sent to - {email_to}")
    except Exception as e:
        print("Error sending email: ", e)

# Capture video from webcam or file
video = cv2.VideoCapture("car-fire-2.mp4")

while True:
    # Read a frame from the video
    ret, frame = video.read()
    if not ret:
        break


    # Resize the frame for better performance
    frame = cv2.resize(frame, (800, 600))

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the "fire" color in HSV
    lower = np.array([0, 70, 50])
    upper = np.array([10, 255, 255])

    # Create a mask to filter out the "fire" color from the frame
    mask = cv2.inRange(hsv, lower, upper)

    # Apply a blur to the mask to remove noise
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Count the number of non-zero pixels in the mask
    size = cv2.countNonZero(mask)

    if size > 12000:
      
        # Fire detected
        if not fire_detected:
            # Play an alarm sound
            fire_detected = True
            cv2.imwrite("fire.jpg", frame)
            send_email("fire.jpg")
           
            email_sent = True
            print("Fire detected")
    else:
        # Fire not detected
        if fire_detected:
            fire_detected = False
            email_sent = False
            print("Fire extinguished")
        

    # Show the mask and the original frame
    cv2.imshow("Mask", mask)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
video.release()
cv2.destroyAllWindows()
