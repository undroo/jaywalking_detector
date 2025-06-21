import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from camera.camera_feed import CameraFeed
import time


class TrafficLightDetection:
    """
    A class to detect traffic lights (red and green) from camera feed.
    
    This class uses color-based detection in HSV color space to identify
    red and green traffic lights in video frames.
    """
    
    def __init__(self, camera_feed: CameraFeed, 
                 red_lower: Tuple[int, int, int] = (0, 100, 100),
                 red_upper: Tuple[int, int, int] = (10, 255, 255),
                 red_lower2: Tuple[int, int, int] = (160, 100, 100),
                 red_upper2: Tuple[int, int, int] = (180, 255, 255),
                 green_lower: Tuple[int, int, int] = (35, 100, 100),
                 green_upper: Tuple[int, int, int] = (85, 255, 255),
                 min_contour_area: int = 1000,
                 confidence_threshold: float = 0.6):
        """
        Initialize the traffic light detector.
        
        Args:
            camera_feed: CameraFeed instance to get frames from
            red_lower: Lower HSV bounds for red color (first range)
            red_upper: Upper HSV bounds for red color (first range)
            red_lower2: Lower HSV bounds for red color (second range, for hue wrap-around)
            red_upper2: Upper HSV bounds for red color (second range, for hue wrap-around)
            green_lower: Lower HSV bounds for green color
            green_upper: Upper HSV bounds for green color
            min_contour_area: Minimum area for a contour to be considered a traffic light
            confidence_threshold: Minimum confidence to consider a detection valid
        """
        self.camera_feed = camera_feed
        self.red_lower = np.array(red_lower)
        self.red_upper = np.array(red_upper)
        self.red_lower2 = np.array(red_lower2)
        self.red_upper2 = np.array(red_upper2)
        self.green_lower = np.array(green_lower)
        self.green_upper = np.array(green_upper)
        self.min_contour_area = min_contour_area
        self.confidence_threshold = confidence_threshold
        
        # Detection history for smoothing
        self.detection_history = []
        self.history_size = 5
        
    def detect_colors(self, frame: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Detect red and green colors in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with 'red' and 'green' contours
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color (two ranges due to hue wrap-around)
        red_mask1 = cv2.inRange(hsv, self.red_lower, self.red_upper)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Create mask for green color
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        red_contours = [c for c in red_contours if cv2.contourArea(c) > self.min_contour_area]
        green_contours = [c for c in green_contours if cv2.contourArea(c) > self.min_contour_area]
        
        return {
            'red': red_contours,
            'green': green_contours
        }
    
    def analyze_traffic_light_shape(self, contours: List[np.ndarray]) -> List[Dict]:
        """
        Analyze contours to determine if they match traffic light characteristics.
        
        Args:
            contours: List of contours to analyze
            
        Returns:
            List of dictionaries with contour info and confidence scores
        """
        traffic_lights = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Approximate contour to polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Traffic light characteristics:
            # - Should be roughly circular (high circularity)
            # - Should have reasonable aspect ratio (not too elongated)
            # - Should have minimum area
            confidence = 0.0
            
            if area > self.min_contour_area:
                # Circularity score (0-1)
                circularity_score = min(circularity, 1.0)
                
                # Aspect ratio score (prefer closer to 1.0)
                aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 0.5) / 0.5
                
                # Area score (prefer larger areas up to a point)
                area_score = min(area / 5000, 1.0)
                
                # Combined confidence
                confidence = (circularity_score * 0.5 + aspect_score * 0.3 + area_score * 0.2)
            
            if confidence > self.confidence_threshold:
                traffic_lights.append({
                    'contour': contour,
                    'confidence': confidence,
                    'area': area,
                    'center': (x + w//2, y + h//2),
                    'bbox': (x, y, w, h)
                })
        
        return traffic_lights
    
    def detect_traffic_lights(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect traffic lights in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with detected red and green traffic lights
        """
        # Detect colors
        color_contours = self.detect_colors(frame)
        
        # Analyze shapes for each color
        red_lights = self.analyze_traffic_light_shape(color_contours['red'])
        green_lights = self.analyze_traffic_light_shape(color_contours['green'])
        
        return {
            'red': red_lights,
            'green': green_lights
        }
    
    def get_traffic_light_state(self, frame: np.ndarray) -> str:
        """
        Determine the current traffic light state.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            String indicating traffic light state: 'red', 'green', or 'unknown'
        """
        detections = self.detect_traffic_lights(frame)
        
        # Get the highest confidence detection for each color
        best_red = max(detections['red'], key=lambda x: x['confidence']) if detections['red'] else None
        best_green = max(detections['green'], key=lambda x: x['confidence']) if detections['green'] else None
        
        # Determine state based on confidence
        red_confidence = best_red['confidence'] if best_red else 0
        green_confidence = best_green['confidence'] if best_green else 0
        
        # Add to history for smoothing
        self.detection_history.append({
            'red_confidence': red_confidence,
            'green_confidence': green_confidence,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Use average confidence from history for more stable detection
        if len(self.detection_history) >= 3:
            avg_red = sum(d['red_confidence'] for d in self.detection_history) / len(self.detection_history)
            avg_green = sum(d['green_confidence'] for d in self.detection_history) / len(self.detection_history)
            
            # Determine state with hysteresis
            if avg_red > avg_green and avg_red > self.confidence_threshold:
                return 'red'
            elif avg_green > avg_red and avg_green > self.confidence_threshold:
                return 'green'
        
        return 'unknown'
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[str, List[Dict]]) -> np.ndarray:
        """
        Draw detection results on the frame.
        
        Args:
            frame: Input frame
            detections: Detection results from detect_traffic_lights
            
        Returns:
            Frame with detections drawn
        """
        result_frame = frame.copy()
        
        # Draw red light detections
        for light in detections['red']:
            x, y, w, h = light['bbox']
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_frame, f"Red: {light['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw green light detections
        for light in detections['green']:
            x, y, w, h = light['bbox']
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_frame, f"Green: {light['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_frame
    
    def run_detection_loop(self, display: bool = True):
        """
        Run continuous traffic light detection.
        
        Args:
            display: Whether to display the video feed with detections
        """
        if not self.camera_feed.start():
            print("Failed to start camera")
            return
        
        print("Traffic Light Detection Started")
        print("Press 'q' to quit")
        
        try:
            while True:
                frame = self.camera_feed.get_frame()
                
                if frame is None:
                    print("Failed to get frame")
                    break
                
                # Detect traffic lights
                detections = self.detect_traffic_lights(frame)
                state = self.get_traffic_light_state(frame)
                
                # Print current state
                print(f"\rTraffic Light State: {state.upper()}", end="", flush=True)
                
                if display:
                    # Draw detections on frame
                    result_frame = self.draw_detections(frame, detections)
                    
                    # Add state text
                    cv2.putText(result_frame, f"State: {state.upper()}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display frame
                    cv2.imshow('Traffic Light Detection', result_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.camera_feed.stop()
            if display:
                cv2.destroyAllWindows()
            print("\nTraffic Light Detection Stopped")


def main():
    """
    Example usage of the TrafficLightDetection class.
    """
    # Create camera feed
    camera = CameraFeed()
    
    # Create traffic light detector
    detector = TrafficLightDetection(camera)
    
    # Run detection
    detector.run_detection_loop()


if __name__ == "__main__":
    main() 