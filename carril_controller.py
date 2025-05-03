from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3].copy()  # eliminar canal alfa/hacer copia editable

def display_image(display, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgba = np.dstack((image_rgb, np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255))
    image_ref = display.imageNew(
        image_rgba.tobytes(),
        Display.RGBA,
        width=image.shape[1],
        height=image.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 15

def set_speed(kmh):
    global speed
    speed = kmh

def set_steering_angle(wheel_angle):
    global angle, steering_angle
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
    angle = max(-0.5, min(0.5, wheel_angle))

def change_steer_angle(inc):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

def follow_lane_center(image):
    height, width, _ = image.shape
    roi = image[int(height * 0.55):height, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    edges_yellow = cv2.Canny(mask_yellow, 50, 150)
    edges_white = cv2.Canny(mask_white, 50, 150)
    
    

    lines_yellow = cv2.HoughLinesP(edges_yellow, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)
    lines_white = cv2.HoughLinesP(edges_white, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)
    
    if lines_yellow is not None and len(lines_yellow) > 10:
        print("[CRUCE] Muchas líneas amarillas detectadas — ir recto")
        return 0.0

    lane_centers = []

    def get_line_center(lines, color, img_offset):
        centers = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2) + img_offset
                centers.append(x_center)
                cv2.line(image, (x1, y1 + img_offset), (x2, y2 + img_offset), color, 2)
                cv2.circle(image, (x_center, y_center), 5, (0, 255, 255), -1)
            avg = int(np.mean(centers))
            lane_centers.append(avg)

    get_line_center(lines_yellow, (0, 255, 255), int(height * 0.6))
    get_line_center(lines_white, (255, 255, 255), int(height * 0.6))

    if len(lane_centers) >= 2:
        lane_center = int(np.mean(lane_centers))
    elif len(lane_centers) == 1:
        lane_center = lane_centers[0]
    else:
        print("[LANE] Líneas no detectadas — seguir recto")
        return 0.0

    cv2.circle(image, (lane_center, height - 10), 8, (0, 0, 255), -1)
    error = (width // 2) - lane_center
    angle = -error / (width // 2) * 0.5
    print(f"[LANE] lane_center: {lane_center}, angle: {angle:.2f}")
    return angle

def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        image = get_image(camera)
        key = keyboard.getKey()

        if key == keyboard.UP:
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN:
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT:
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT:
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)

        angle = follow_lane_center(image)
        display_image(display_img, image)

        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
