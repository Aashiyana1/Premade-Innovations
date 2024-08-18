import threading
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# load the COCO class names
with open('Model/COCO_labels.txt', 'r') as f:
    class_names = f.read().split('\n')


# Charger le modèle de détection d'objets de OpenCV
net =cv2.dnn.readNet(model='Model/frozen_inference_graph_V2.pb',
                        config='Model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt', 
                        framework='TensorFlow')

def select_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    # Load the selected image
    image = cv2.imread(file_path)
    # Perform the people detection and counting
    # Obtenir les dimensions de l'image
    (H, W) = image.shape[:2]

    # Construire le blob à partir de l'image et effectuer la détection d'objets
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Initialiser le compteur de personnes
    count = 0

    # Boucle sur chaque détection
    for i in range(0, detections.shape[2]):
        # Extraire la confiance (probabilité) de la détection 
        confidence = detections[0, 0, i, 2]

        # Filtrer les détections faibles
        if confidence > 0.4:

            # get the class id
            class_id = detections[0,0,i,1]
                # map the class id to the class 
            class_name = class_names[int(class_id)-1]
            # Calculer les coordonnées (x, y) de la boîte de détection en utilisant ses dimensions et sa position centrale
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
        
            if class_name == 'person':
                # Dessiner la boîte de détection et afficher l'étiquette et la confiance
                label = "{}: {:.2f}%".format("person", confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Mettre à jour le compteur de personnes
                count += 1

    cv2.putText(image, f"{count} persons", (W-200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0 ), 2)
    # Display the result in the GUI
    # cv2.imshow("Result", image)
    display_image(image)
    cv2.imwrite('ressources/image_result.jpg', image)
    # Hide the stop button
    stop_button.pack_forget()

def select_video():
    # Reset the stop flag
    stop_flag.clear()
    # Open a file dialog to select a video
    file_path = filedialog.askopenfilename()
    # Load the selected video
    video = cv2.VideoCapture(file_path)
    # Start a separate thread to display the video frames
    video_thread = threading.Thread(target=display_video, args=(video,))
    video_thread.start()

    # Show the stop button
    stop_button.pack()
    
def display_video(video):

    if not video.isOpened():
        print("Error opening video file")

    # get the video frames' width and height for proper saving of videos
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    # create the VideoWriter() object
    out = cv2.VideoWriter('ressources/video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                        (frame_width, frame_height))

    # Read and process each frame
    while True:

        # Check if the stop flag is set
        if stop_flag.is_set():
            break
        # Read the frame
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            break

        # Get the dimensions of the frame
        (H, W) = frame.shape[:2]

        # Construct a blob from the frame and perform object detection
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Initialize the person count
        count = 0

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (probability) of the detection
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.4:

                # get the class id
                class_id = detections[0,0,i,1]
                # map the class id to the class 
                class_name = class_names[int(class_id)-1]
            
                # Extract the class label and bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                # Increment the person count
                if class_name == 'person':
                    count += 1
                        # Draw a rectangle around the person
                    label = "{}: {:.2f}%".format("person", confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.putText(frame, f"{count} persons", (W-200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255 ), 2)
        
        # Display the frame
        cv2.imshow("Output", frame)
        # display_image(frame)
        out.write(frame) 
    
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    # Release the resources
    cv2.destroyAllWindows()
    video.release()
    out.release()

    # Hide the stop button
    stop_button.pack_forget()
    image_label.config(image='')

def display_image(image):

    # Resize the image to fit the size of the image label
    width, height = image_label.winfo_width(), image_label.winfo_height()
    if width > 0 and height > 0:
        image = cv2.resize(image, (width, height))
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image from NumPy array to PhotoImage
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    # Update the image label with the new image
    image_label.config(image=image)
    image_label.image = image

    image_label.config(image=image)
    image_label.image = image
    
def select_camera():

    stop_button.pack_forget()
    video = cv2.VideoCapture(0)
        # get the video frames' width and height for proper saving of videos
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    # create the VideoWriter() object
    out = cv2.VideoWriter('ressources/Camera_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                        (frame_width, frame_height))

    # Read and process each frame
    while True:
        # Read the frame
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            break

        # Get the dimensions of the frame
        (H, W) = frame.shape[:2]

        # Construct a blob from the frame and perform object detection
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Initialize the person count
        count = 0

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (probability) of the detection
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.4:

                # get the class id
                class_id = detections[0,0,i,1]
                # map the class id to the class 
                class_name = class_names[int(class_id)-1]
            
                # Extract the class label and bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])

                (startX, startY, endX, endY) = box.astype("int")
                # Increment the person count
                if class_name == 'person':
                    count += 1
                        # Draw a rectangle around the person
                    label = "{}: {:.2f}%".format("person", confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        cv2.putText(frame, f"{count} persons", (W-200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255 ), 2)
        
        # Display the frame
        cv2.imshow("Output", frame)
        out.write(frame) 
    
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    # Release the resources
    cv2.destroyAllWindows()
    video.release()
    out.release()

def stop_video():
    # Set the stop flag to stop the video display
    stop_flag.set()
   

root = tk.Tk()
root.title("People Detection and Counting")
root.geometry("1000x600")

image_label = tk.Label(root)
image_label.pack(fill="both", expand=True)

buttons_frame = tk.Frame(root)
buttons_frame.pack(side="top", fill="y")

video_button = tk.Button(buttons_frame, text="Select Video", command=select_video)
video_button.pack(side="left")

camera_button = tk.Button(buttons_frame, text="Select Camera", command=select_camera)
camera_button.pack(side="left")

stop_button = tk.Button(buttons_frame, text="Stop Video", command=stop_video)
stop_button.pack(side = "left")

stop_flag = threading.Event()
stop_button.pack_forget()

root.mainloop()