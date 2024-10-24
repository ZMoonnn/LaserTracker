
# Laser Tracker using OpenCV

This project tracks a laser pointer using OpenCV and a USB or laptop webcam. The system detects the laser's position based on HSV color thresholds and outputs its coordinates.

## Features
- Real-time laser pointer tracking.
- Adjustable HSV thresholds via GUI trackbars.
- Option to display the thresholded image for fine-tuning.
- Simple, user-friendly Python script that can be easily extended.

## Requirements
- Python 3.x
- OpenCV 4.x
- NumPy

## Quick Start

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy
   ```

3. Run the script:
   ```bash
   python track_laser.py
   ```

## Adjusting HSV Values

You can adjust the HSV values to match the color of your laser pointer. Use the trackbars that appear when you run the script to fine-tune the detection.

## Example

The script outputs the coordinates of the laser in real-time. If no laser is detected, it will indicate so in the console.

## License

This project is open-source and available under the MIT License.

