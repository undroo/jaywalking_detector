from camera.camera_feed import CameraFeed
from models.slow_walker_detection import SlowWalkerDetection
import cv2
import time

def main():
    """
    Main function running slow walker detection with roasts.
    """
    print("Slow Walker Detection with Roasts")
    print("Starting detection system...")
    
    # Run slow walker detection
    run_slow_walker_detection()

def run_slow_walker_detection():
    """
    Run the slow walker detection system with roasts.
    """
    # Create camera feed
    camera = CameraFeed()
    
    # Create slow walker detector
    detector = SlowWalkerDetection(camera)
    
    # Run detection
    detector.run_detection_loop()

if __name__ == "__main__":
    main()
