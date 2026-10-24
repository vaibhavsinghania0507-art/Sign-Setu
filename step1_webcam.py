import cv2

# Open the webcam. 0 means "the first camera", your laptop's built-in one.
camera = cv2.VideoCapture(0)

while True:
    # Grab one frame. read() gives back two things:
    # success = True/False (did it work?), frame = the picture itself
    success, frame = camera.read()

    # If the camera failed (unplugged, in use by another app), stop the loop
    if not success:
        print("Couldn't read from the webcam")
        break

    # Mirror the picture so it acts like a mirror when you sign
    frame = cv2.flip(frame, 1)

    # Show the frame in a window called "SignSetu"
    cv2.imshow("SignSetu", frame)

    # Wait 1 millisecond for a key press. If it's 'q', leave the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Give the camera back to the system and close the window
camera.release()
cv2.destroyAllWindows()