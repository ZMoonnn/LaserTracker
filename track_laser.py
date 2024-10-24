import argparse
import cv2
import sys
import numpy as np

class LaserTracker(object):
    def __init__(self, cam_width=640, cam_height=480, hue_min=5, hue_max=6,
                 sat_min=50, sat_max=100, val_min=250, val_max=256,
                 display_thresholds=False):
        # Initialize camera settings and HSV thresholds.
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.display_thresholds = display_thresholds  # Toggle display of thresholded images.
        self.capture = None  # Camera capture device instance.
        self.distance_limit = 2.0  # Maximum tracking distance in meters.
        self.tolerance_px = 32  # Pixel tolerance for detection accuracy.

    def create_and_position_window(self, name, xpos, ypos):
        # Create and position a window with specified name and coordinates.
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(name, self.cam_width, self.cam_height)
        cv2.moveWindow(name, xpos, ypos)

    def setup_camera_capture(self, device_num=0):
        # Initialize camera capture with specified device number.
        try:
            device = int(device_num)
            print(f"Using Camera Device: {device}")
        except (IndexError, ValueError):
            device = 0
            print("Invalid Device. Using default device 0", file=sys.stderr)
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            print("Failed to open capture device. Exiting.", file=sys.stderr)
            self.clean_exit()

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)  # Set camera width.
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)  # Set camera height.

    def handle_quit(self, delay=10):
        # Check for quit command with specified delay.
        key = cv2.waitKey(delay)
        return key == 27 or key in [ord('q'), ord('Q')]  # True if 'q' or ESC pressed.

    def clean_exit(self):
        # Release resources and exit cleanly.
        if self.capture and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    def create_hsv_trackbars(self):
        # Create HSV adjustment trackbars for real-time tweaking.
        cv2.namedWindow("HSV Trackbars")
        cv2.createTrackbar("Hue Min", "HSV Trackbars", self.hue_min, 179, self.nothing)
        cv2.createTrackbar("Hue Max", "HSV Trackbars", self.hue_max, 179, self.nothing)
        cv2.createTrackbar("Sat Min", "HSV Trackbars", self.sat_min, 255, self.nothing)
        cv2.createTrackbar("Sat Max", "HSV Trackbars", self.sat_max, 255, self.nothing)
        cv2.createTrackbar("Val Min", "HSV Trackbars", self.val_min, 255, self.nothing)
        cv2.createTrackbar("Val Max", "HSV Trackbars", self.val_max, 255, self.nothing)

    def get_trackbar_values(self):
        # Retrieve current settings from HSV trackbars.
        self.hue_min = cv2.getTrackbarPos("Hue Min", "HSV Trackbars")
        self.hue_max = cv2.getTrackbarPos("Hue Max", "HSV Trackbars")
        self.sat_min = cv2.getTrackbarPos("Sat Min", "HSV Trackbars")
        self.sat_max = cv2.getTrackbarPos("Sat Max", "HSV Trackbars")
        self.val_min = cv2.getTrackbarPos("Val Min", "HSV Trackbars")
        self.val_max = cv2.getTrackbarPos("Val Max", "HSV Trackbars")

    def nothing(self, x):
        # Placeholder function for trackbar interaction.
        pass

    def estimate_distance(self, contour_area):
        # Estimate distance to the laser based on its contour area.
        if contour_area > 0:
            return 1.0 / (contour_area ** 0.5)  # Inverse square law approximation.

    def detect(self, frame):
        # Process frame to detect laser based on current HSV settings.
        self.get_trackbar_values()  # Update HSV values based on trackbar positions.
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Reduce noise.
        hsv_img = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)  # Convert to HSV.

        LASER_MIN = np.array([self.hue_min, self.sat_min, self.val_min], np.uint8)  # Min HSV threshold.
        LASER_MAX = np.array([self.hue_max, self.sat_max, self.val_max], np.uint8)  # Max HSV threshold.
        frame_threshed = cv2.inRange(hsv_img, LASER_MIN, LASER_MAX)  # Apply HSV threshold.

        kernel = np.ones((5, 5), np.uint8)  # Morphological operations to clean image.
        frame_threshed = cv2.morphologyEx(frame_threshed, cv2.MORPH_OPEN, kernel)
        frame_threshed = cv2.morphologyEx(frame_threshed, cv2.MORPH_CLOSE, kernel)

        if self.display_thresholds:  # Optionally display the thresholded image.
            cv2.imshow("Thresholded Image", frame_threshed)

        contours, _ = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours.
        laserx, lasery = 0, 0  # Initialize coordinates.
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            if contour_area > 100:  # Filter out small noise.
                distance = self.estimate_distance(contour_area)
                if distance and distance <= self.distance_limit:  # Check distance limit.
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:  # Calculate centroid if area is non-zero.
                        laserx = int(M["m10"] / M["m00"])
                        lasery = int(M["m01"] / M["m00"])
                    if self.is_within_tolerance(laserx, lasery):  # Check tolerance.
                        return laserx, lasery
                else:
                    print(f"Laser detected but too far: {distance:.2f} meters")
        return laserx, lasery  # Return coordinates of laser or (0,0) if not found.

    def is_within_tolerance(self, laserx, lasery):
        # Check if detected laser is within defined tolerance.
        return True  # Placeholder for actual tolerance checking logic.

    def display(self, frame):
        # Display the current video frame.
        cv2.imshow('RGB_VideoFrame', frame)

    def run(self):
        # Main loop to run the laser tracking process.
        print(f"Using OpenCV version: {cv2.__version__}")
        self.create_and_position_window('RGB_VideoFrame', 10 + self.cam_width, 0)
        self.create_hsv_trackbars()
        self.setup_camera_capture(device_num=0)

        while True:
            success, frame = self.capture.read()  # Read a frame from the camera.
            if not success:
                print("Could not read camera frame. Exiting.", file=sys.stderr)
                break

            laserx, lasery = self.detect(frame)  # Detect laser in the frame.
            if laserx == 0 and lasery == 0:
                print("No laser detected")
            else:
                print(f"Laser detected at: ({laserx}, {lasery})")

            self.display(frame)  # Display the frame.

            if self.handle_quit():  # Check for quit condition.
                break

        self.clean_exit()  # Clean up resources on exit.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Laser Tracker')
    parser.add_argument('-W', '--width', default=640, type=int, help='Camera Width')
    parser.add_argument('-H', '--height', default=480, type=int, help='Camera Height')
    parser.add_argument('-u', '--huemin', default=5, type=int, help='Hue Minimum Threshold')
    parser.add_argument('-U', '--huemax', default=6, type=int, help='Hue Maximum Threshold')
    parser.add_argument('-s', '--satmin', default=50, type=int, help='Saturation Minimum Threshold')
    parser.add_argument('-S', '--satmax', default=100, type=int, help='Saturation Maximum Threshold')
    parser.add_argument('-v', '--valmin', default=250, type=int, help='Value Minimum Threshold')
    parser.add_argument('-V', '--valmax', default=256, type=int, help='Value Maximum Threshold')
    parser.add_argument('-d', '--display', action='store_true', help='Display Threshold Windows')
    params = parser.parse_args()

    tracker = LaserTracker(
        cam_width=params.width,
        cam_height=params.height,
        hue_min=params.huemin,
        hue_max=params.huemax,
        sat_min=params.satmin,
        sat_max=params.satmax,
        val_min=params.valmin,
        val_max=params.valmax,
        display_thresholds=params.display
    )
    tracker.run()
