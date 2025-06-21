import cv2
import numpy as np
from typing import Optional, Tuple
import time


class CameraFeed:
    """
    A class to handle webcam video feed using OpenCV.
    
    This class provides functionality to:
    - Initialize and connect to the laptop webcam
    - Start and stop video capture
    - Get individual frames from the camera
    - Release camera resources properly
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the camera feed.
        
        Args:
            camera_index (int): Index of the camera device (0 for default laptop webcam)
            width (int): Desired frame width
            height (int): Desired frame height
            fps (int): Desired frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        
    def start(self) -> bool:
        """
        Start the camera feed.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            print(f"Camera started successfully at {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera feed and release resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a single frame from the camera.
        
        Returns:
            np.ndarray: Frame as a numpy array, or None if frame couldn't be read
        """
        if not self.is_running or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Error: Could not read frame from camera")
            return None
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the current frame size.
        
        Returns:
            Tuple[int, int]: (width, height) of the frame
        """
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (self.width, self.height)
    
    def is_camera_available(self) -> bool:
        """
        Check if the camera is available and working.
        
        Returns:
            bool: True if camera is available, False otherwise
        """
        if self.cap is None:
            return False
        return self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


