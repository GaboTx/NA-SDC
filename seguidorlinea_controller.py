from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # Drop alpha channel

# Image processing
def find_steering_direction(image_bgr):
    height, width = image_bgr.shape[:2]
    
    # 1. Region of Interest: lower half
    roi = image_bgr[int(height * 0.6):, :]

    # 2. Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 3. Yellow mask
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 4. Edges
    edges = cv2.Canny(mask, 50, 150)

    # 5. Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=50)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        if angles:
            avg_angle = np.mean(angles)
            print(f"[DETECTED] Angle: {avg_angle:.2f} rad")
            return avg_angle
    print("[NO LINE] — going straight")
    return 0.0

# Display image 
def display_image(display, image):
    image_rgb = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Global steering and speed values
steering_angle = 0
speed = 15

def set_speed(kmh):
    global speed
    speed = kmh

def set_steering_angle(wheel_angle):
    global steering_angle
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = max(min(wheel_angle, 0.5), -0.5)

# MAIN FUNCTION
def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    # Devices
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        image = get_image(camera)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_image(display_img, grey_image)

        # KEYBOARD MODE
        key = keyboard.getKey()
        if key == Keyboard.UP:
            set_speed(speed + 5)
        elif key == Keyboard.DOWN:
            set_speed(speed - 5)
        elif key == Keyboard.RIGHT:
            set_steering_angle(steering_angle + 0.1)
        elif key == Keyboard.LEFT:
            set_steering_angle(steering_angle - 0.1)
        elif key == ord('A'):
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
        else:
            # AUTO MODE if no arrow pressed
            lane_angle = find_steering_direction(image)
            set_steering_angle(lane_angle)

        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
