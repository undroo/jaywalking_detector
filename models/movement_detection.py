import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from camera.camera_feed import CameraFeed
import time


class MovementDetection:
    """
    A class to detect movement from camera feed.
    
    This class uses background subtraction and motion analysis to identify
    moving objects in video frames. It can detect both general movement
    and track specific moving objects.
    """
    
    def __init__(self, camera_feed: CameraFeed,
                 history_length: int = 50,
                 var_threshold: float = 16.0,
                 detect_shadows: bool = True,
                 min_contour_area: int = 500,
                 movement_threshold: float = 0.01,
                 smoothing_frames: int = 5):
        """
        Initialize the movement detector.
        
        Args:
            camera_feed: CameraFeed instance to get frames from
            history_length: Number of frames to use for background model
            var_threshold: Threshold for pixel-wise segmentation
            detect_shadows: Whether to detect and mark shadows
            min_contour_area: Minimum area for a contour to be considered movement
            movement_threshold: Minimum percentage of frame that must be moving
            smoothing_frames: Number of frames to average for smoothing
        """
        self.camera_feed = camera_feed
        self.history_length = history_length
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_contour_area = min_contour_area
        self.movement_threshold = movement_threshold
        self.smoothing_frames = smoothing_frames
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history_length,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Movement history for smoothing
        self.movement_history = []
        self.frame_count = 0
        
        # Store previous frame for frame differencing
        self.prev_frame = None
        
    def detect_movement_background_subtraction(self, frame: np.ndarray) -> Dict:
        """
        Detect movement using background subtraction.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with movement detection results
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (set to 0)
        if self.detect_shadows:
            fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Calculate movement metrics
        total_pixels = frame.shape[0] * frame.shape[1]
        moving_pixels = np.sum(fg_mask > 0)
        movement_percentage = moving_pixels / total_pixels
        
        return {
            'contours': valid_contours,
            'mask': fg_mask,
            'movement_percentage': movement_percentage,
            'moving_pixels': moving_pixels,
            'total_pixels': total_pixels
        }
    
    def detect_movement_frame_differencing(self, frame: np.ndarray) -> Dict:
        """
        Detect movement using frame differencing.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with movement detection results
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return {
                'contours': [],
                'mask': np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                'movement_percentage': 0.0,
                'moving_pixels': 0,
                'total_pixels': frame.shape[0] * frame.shape[1]
            }
        
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Calculate movement metrics
        total_pixels = frame.shape[0] * frame.shape[1]
        moving_pixels = np.sum(thresh > 0)
        movement_percentage = moving_pixels / total_pixels
        
        # Update previous frame
        self.prev_frame = gray
        
        return {
            'contours': valid_contours,
            'mask': thresh,
            'movement_percentage': movement_percentage,
            'moving_pixels': moving_pixels,
            'total_pixels': total_pixels
        }
    
    def detect_movement(self, frame: np.ndarray, method: str = 'background') -> Dict:
        """
        Detect movement using specified method.
        
        Args:
            frame: Input frame as numpy array
            method: Detection method ('background' or 'frame_diff')
            
        Returns:
            Dictionary with movement detection results
        """
        if method == 'background':
            return self.detect_movement_background_subtraction(frame)
        elif method == 'frame_diff':
            return self.detect_movement_frame_differencing(frame)
        else:
            raise ValueError("Method must be 'background' or 'frame_diff'")
    
    def is_movement_detected(self, frame: np.ndarray, method: str = 'background') -> bool:
        """
        Determine if significant movement is detected.
        
        Args:
            frame: Input frame as numpy array
            method: Detection method ('background' or 'frame_diff')
            
        Returns:
            True if movement is detected, False otherwise
        """
        detection = self.detect_movement(frame, method)
        
        # Add to history for smoothing
        self.movement_history.append({
            'movement_percentage': detection['movement_percentage'],
            'contour_count': len(detection['contours']),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.movement_history) > self.smoothing_frames:
            self.movement_history.pop(0)
        
        # Use average movement percentage for more stable detection
        if len(self.movement_history) >= 3:
            avg_movement = sum(d['movement_percentage'] for d in self.movement_history) / len(self.movement_history)
            return avg_movement > self.movement_threshold
        
        return detection['movement_percentage'] > self.movement_threshold
    
    def get_movement_metrics(self, frame: np.ndarray, method: str = 'background') -> Dict:
        """
        Get detailed movement metrics.
        
        Args:
            frame: Input frame as numpy array
            method: Detection method ('background' or 'frame_diff')
            
        Returns:
            Dictionary with movement metrics
        """
        detection = self.detect_movement(frame, method)
        
        # Calculate additional metrics
        metrics = {
            'movement_percentage': detection['movement_percentage'],
            'moving_pixels': detection['moving_pixels'],
            'total_pixels': detection['total_pixels'],
            'contour_count': len(detection['contours']),
            'is_moving': detection['movement_percentage'] > self.movement_threshold
        }
        
        # Calculate contour areas and centers
        if detection['contours']:
            areas = [cv2.contourArea(c) for c in detection['contours']]
            centers = []
            for contour in detection['contours']:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
                else:
                    centers.append((0, 0))
            
            metrics.update({
                'largest_contour_area': max(areas),
                'average_contour_area': sum(areas) / len(areas),
                'contour_centers': centers
            })
        else:
            metrics.update({
                'largest_contour_area': 0,
                'average_contour_area': 0,
                'contour_centers': []
            })
        
        return metrics
    
    def draw_movement(self, frame: np.ndarray, detection: Dict, 
                     draw_contours: bool = True, draw_mask: bool = False) -> np.ndarray:
        """
        Draw movement detection results on the frame.
        
        Args:
            frame: Input frame
            detection: Detection results from detect_movement
            draw_contours: Whether to draw bounding boxes around moving objects
            draw_mask: Whether to overlay the movement mask
            
        Returns:
            Frame with movement visualization
        """
        result_frame = frame.copy()
        
        # Overlay movement mask if requested
        if draw_mask and 'mask' in detection:
            mask_colored = cv2.cvtColor(detection['mask'], cv2.COLOR_GRAY2BGR)
            # Make mask semi-transparent
            result_frame = cv2.addWeighted(result_frame, 0.7, mask_colored, 0.3, 0)
        
        # Draw contours and bounding boxes
        if draw_contours:
            for i, contour in enumerate(detection['contours']):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate contour area
                area = cv2.contourArea(contour)
                
                # Draw bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add contour number and area
                cv2.putText(result_frame, f"#{i+1}: {area:.0f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add movement percentage text
        movement_text = f"Movement: {detection['movement_percentage']:.2%}"
        cv2.putText(result_frame, movement_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add contour count
        contour_text = f"Objects: {len(detection['contours'])}"
        cv2.putText(result_frame, contour_text, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result_frame
    
    def run_detection_loop(self, method: str = 'background', display: bool = True):
        """
        Run continuous movement detection.
        
        Args:
            method: Detection method ('background' or 'frame_diff')
            display: Whether to display the video feed with detections
        """
        if not self.camera_feed.start():
            print("Failed to start camera")
            return
        
        print(f"Movement Detection Started (Method: {method})")
        print("Press 'q' to quit, 'm' to toggle mask overlay")
        
        show_mask = False
        
        try:
            while True:
                frame = self.camera_feed.get_frame()
                
                if frame is None:
                    print("Failed to get frame")
                    break
                
                # Detect movement
                detection = self.detect_movement(frame, method)
                is_moving = self.is_movement_detected(frame, method)
                
                # Print current status
                status = "MOVING" if is_moving else "STILL"
                print(f"\rStatus: {status} | Movement: {detection['movement_percentage']:.2%} | Objects: {len(detection['contours'])}", 
                      end="", flush=True)
                
                if display:
                    # Draw detections on frame
                    result_frame = self.draw_movement(frame, detection, draw_mask=show_mask)
                    
                    # Add status indicator
                    color = (0, 255, 0) if is_moving else (0, 0, 255)
                    cv2.putText(result_frame, f"STATUS: {status}", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Display frame
                    cv2.imshow('Movement Detection', result_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('m'):
                        show_mask = not show_mask
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.camera_feed.stop()
            if display:
                cv2.destroyAllWindows()
            print("\nMovement Detection Stopped")


def main():
    """
    Example usage of the MovementDetection class.
    """
    # Create camera feed
    camera = CameraFeed()
    
    # Create movement detector
    detector = MovementDetection(camera)
    
    # Run detection
    detector.run_detection_loop()


if __name__ == "__main__":
    main()
