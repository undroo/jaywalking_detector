import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from camera.camera_feed import CameraFeed
from models.traffic_light_detection import TrafficLightDetection
from models.movement_detection import MovementDetection
import time


class JaywalkingDetection:
    """
    A class to detect jaywalking behavior by combining traffic light and movement detection.
    
    This class monitors for movement when traffic lights are red, which indicates
    potential jaywalking behavior.
    """
    
    def __init__(self, camera_feed: CameraFeed,
                 confidence_threshold: float = 0.6,
                 movement_threshold: float = 0.01,
                 jaywalking_duration_threshold: float = 1.0,
                 alert_cooldown: float = 3.0):
        """
        Initialize the jaywalking detector.
        
        Args:
            camera_feed: CameraFeed instance to get frames from
            confidence_threshold: Minimum confidence for traffic light detection
            movement_threshold: Minimum movement percentage to consider significant
            jaywalking_duration_threshold: Minimum duration (seconds) to trigger jaywalking alert
            alert_cooldown: Cooldown period (seconds) between alerts
        """
        self.camera_feed = camera_feed
        self.confidence_threshold = confidence_threshold
        self.movement_threshold = movement_threshold
        self.jaywalking_duration_threshold = jaywalking_duration_threshold
        self.alert_cooldown = alert_cooldown
        
        # Initialize detection systems
        self.traffic_detector = TrafficLightDetection(
            camera_feed=camera_feed,
            confidence_threshold=confidence_threshold
        )
        self.movement_detector = MovementDetection(
            camera_feed=camera_feed,
            movement_threshold=movement_threshold
        )
        
        # Jaywalking detection state
        self.jaywalking_start_time = None
        self.last_alert_time = 0
        self.jaywalking_events = []
        self.is_jaywalking = False
        
        # Detection history for smoothing
        self.detection_history = []
        self.history_size = 10
        
    def detect_jaywalking(self, frame: np.ndarray) -> Dict:
        """
        Detect jaywalking behavior in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with jaywalking detection results
        """
        current_time = time.time()
        
        # Detect traffic light state
        traffic_state = self.traffic_detector.get_traffic_light_state(frame)
        traffic_detections = self.traffic_detector.detect_traffic_lights(frame)
        
        # Detect movement
        movement_detection = self.movement_detector.detect_movement(frame)
        is_moving = self.movement_detector.is_movement_detected(frame)
        
        # Determine if conditions for jaywalking are met
        red_light = traffic_state == 'red'
        significant_movement = is_moving
        
        # Add to detection history
        self.detection_history.append({
            'timestamp': current_time,
            'red_light': red_light,
            'significant_movement': significant_movement,
            'traffic_state': traffic_state,
            'movement_percentage': movement_detection['movement_percentage'],
            'traffic_confidence': max([d['confidence'] for d in traffic_detections['red']]) if traffic_detections['red'] else 0
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Analyze jaywalking behavior with temporal smoothing
        jaywalking_detected = self._analyze_jaywalking_behavior()
        
        # Update jaywalking state
        if jaywalking_detected and not self.is_jaywalking:
            # Jaywalking just started
            self.jaywalking_start_time = current_time
            self.is_jaywalking = True
        elif not jaywalking_detected and self.is_jaywalking:
            # Jaywalking stopped
            if self.jaywalking_start_time:
                duration = current_time - self.jaywalking_start_time
                if duration >= self.jaywalking_duration_threshold:
                    # Record jaywalking event
                    self.jaywalking_events.append({
                        'start_time': self.jaywalking_start_time,
                        'end_time': current_time,
                        'duration': duration
                    })
            self.is_jaywalking = False
            self.jaywalking_start_time = None
        
        # Check if we should trigger an alert
        alert_triggered = False
        if (self.is_jaywalking and 
            self.jaywalking_start_time and 
            current_time - self.jaywalking_start_time >= self.jaywalking_duration_threshold and
            current_time - self.last_alert_time >= self.alert_cooldown):
            alert_triggered = True
            self.last_alert_time = current_time
        
        return {
            'jaywalking_detected': jaywalking_detected,
            'is_jaywalking': self.is_jaywalking,
            'alert_triggered': alert_triggered,
            'traffic_state': traffic_state,
            'red_light': red_light,
            'significant_movement': significant_movement,
            'movement_percentage': movement_detection['movement_percentage'],
            'traffic_detections': traffic_detections,
            'movement_detection': movement_detection,
            'jaywalking_duration': current_time - self.jaywalking_start_time if self.jaywalking_start_time else 0,
            'total_jaywalking_events': len(self.jaywalking_events)
        }
    
    def _analyze_jaywalking_behavior(self) -> bool:
        """
        Analyze detection history to determine if jaywalking is occurring.
        
        Returns:
            True if jaywalking is detected, False otherwise
        """
        if len(self.detection_history) < 3:
            return False
        
        # Get recent detections
        recent_detections = self.detection_history[-5:]
        
        # Count red light and movement occurrences
        red_light_count = sum(1 for d in recent_detections if d['red_light'])
        movement_count = sum(1 for d in recent_detections if d['significant_movement'])
        
        # Require both red light and movement for majority of recent frames
        red_light_ratio = red_light_count / len(recent_detections)
        movement_ratio = movement_count / len(recent_detections)
        
        # Both conditions must be met for majority of recent frames
        return red_light_ratio >= 0.6 and movement_ratio >= 0.6
    
    def get_jaywalking_statistics(self) -> Dict:
        """
        Get statistics about detected jaywalking events.
        
        Returns:
            Dictionary with jaywalking statistics
        """
        if not self.jaywalking_events:
            return {
                'total_events': 0,
                'average_duration': 0,
                'longest_duration': 0,
                'shortest_duration': 0
            }
        
        durations = [event['duration'] for event in self.jaywalking_events]
        
        return {
            'total_events': len(self.jaywalking_events),
            'average_duration': sum(durations) / len(durations),
            'longest_duration': max(durations),
            'shortest_duration': min(durations),
            'recent_events': self.jaywalking_events[-5:]  # Last 5 events
        }
    
    def draw_jaywalking_detection(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw jaywalking detection results on the frame.
        
        Args:
            frame: Input frame
            detection: Detection results from detect_jaywalking
            
        Returns:
            Frame with jaywalking visualization
        """
        result_frame = frame.copy()
        
        # Draw traffic light detections
        result_frame = self.traffic_detector.draw_detections(
            result_frame, detection['traffic_detections']
        )
        
        # Draw movement detections
        result_frame = self.movement_detector.draw_movement(
            result_frame, detection['movement_detection']
        )
        
        # Add jaywalking status
        if detection['is_jaywalking']:
            # Red background for jaywalking alert
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (0, 0), (result_frame.shape[1], 150), (0, 0, 255), -1)
            result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)
            
            # Jaywalking alert text
            cv2.putText(result_frame, "JAYWALKING DETECTED!", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Duration
            duration = detection['jaywalking_duration']
            cv2.putText(result_frame, f"Duration: {duration:.1f}s", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Alert indicator
            if detection['alert_triggered']:
                cv2.putText(result_frame, "ALERT TRIGGERED!", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Normal status
            status_color = (0, 255, 0) if detection['traffic_state'] == 'green' else (0, 0, 255)
            cv2.putText(result_frame, f"Traffic: {detection['traffic_state'].upper()}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            movement_status = "MOVING" if detection['significant_movement'] else "STILL"
            movement_color = (0, 255, 0) if detection['significant_movement'] else (128, 128, 128)
            cv2.putText(result_frame, f"Movement: {movement_status}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, movement_color, 2)
        
        # Add statistics
        stats = self.get_jaywalking_statistics()
        cv2.putText(result_frame, f"Total Events: {stats['total_events']}", 
                   (10, result_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if stats['total_events'] > 0:
            cv2.putText(result_frame, f"Avg Duration: {stats['average_duration']:.1f}s", 
                       (10, result_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def run_detection_loop(self, display: bool = True, save_events: bool = True):
        """
        Run continuous jaywalking detection.
        
        Args:
            display: Whether to display the video feed with detections
            save_events: Whether to save jaywalking events to file
        """
        if not self.camera_feed.start():
            print("Failed to start camera")
            return
        
        print("Jaywalking Detection Started")
        print("Press 'q' to quit, 's' to show statistics")
        
        try:
            while True:
                frame = self.camera_feed.get_frame()
                
                if frame is None:
                    print("Failed to get frame")
                    break
                
                # Detect jaywalking
                detection = self.detect_jaywalking(frame)
                
                # Print current status
                status = "JAYWALKING" if detection['is_jaywalking'] else "NORMAL"
                print(f"\rStatus: {status} | Traffic: {detection['traffic_state']} | Movement: {detection['movement_percentage']:.2%} | Events: {detection['total_jaywalking_events']}", 
                      end="", flush=True)
                
                # Handle alert
                if detection['alert_triggered']:
                    print(f"\n🚨 JAYWALKING ALERT! Duration: {detection['jaywalking_duration']:.1f}s")
                    if save_events:
                        self._save_jaywalking_event(detection)
                
                if display:
                    # Draw detections on frame
                    result_frame = self.draw_jaywalking_detection(frame, detection)
                    
                    # Display frame
                    cv2.imshow('Jaywalking Detection', result_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._print_statistics()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.camera_feed.stop()
            if display:
                cv2.destroyAllWindows()
            print("\nJaywalking Detection Stopped")
            
            # Print final statistics
            self._print_statistics()
    
    def _save_jaywalking_event(self, detection: Dict):
        """
        Save jaywalking event to file.
        
        Args:
            detection: Detection results containing jaywalking event
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        duration = detection['jaywalking_duration']
        
        with open("jaywalking_events.log", "a") as f:
            f.write(f"{timestamp} - Jaywalking detected for {duration:.1f} seconds\n")
    
    def _print_statistics(self):
        """
        Print jaywalking detection statistics.
        """
        stats = self.get_jaywalking_statistics()
        print(f"\n=== Jaywalking Detection Statistics ===")
        print(f"Total Events: {stats['total_events']}")
        if stats['total_events'] > 0:
            print(f"Average Duration: {stats['average_duration']:.1f} seconds")
            print(f"Longest Duration: {stats['longest_duration']:.1f} seconds")
            print(f"Shortest Duration: {stats['shortest_duration']:.1f} seconds")
            print(f"Recent Events: {len(stats['recent_events'])}")
        print("=======================================")


def main():
    """
    Example usage of the JaywalkingDetection class.
    """
    # Create camera feed
    camera = CameraFeed()
    
    # Create jaywalking detector
    detector = JaywalkingDetection(camera)
    
    # Run detection
    detector.run_detection_loop()


if __name__ == "__main__":
    main() 