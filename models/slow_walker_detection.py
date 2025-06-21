import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from camera.camera_feed import CameraFeed
from models.movement_detection import MovementDetection
from models.shame_bot import ShameBot
from PIL import Image
import time
import threading



class SlowWalkerDetection:
    """
    A class to detect slow walkers and roast them using the ShameBot.
    
    This class monitors movement patterns to identify people walking slowly
    and generates humorous roasts about their walking speed.
    """
    
    def __init__(self, camera_feed: CameraFeed,
                 slow_movement_threshold: float = 0.05,
                 fast_movement_threshold: float = 0.1,
                 slow_duration_threshold: float = 1.0,
                 roast_cooldown: float = 5.0,
                 enable_voice: bool = True,
                 frame_skip: int = 1,  # Process every Nth frame for speed
                 history_size: int = 5):  # Reduced from 10 for faster response
        """
        Initialize the slow walker detector.
        
        Args:
            camera_feed: CameraFeed instance to get frames from
            slow_movement_threshold: Maximum movement percentage to be considered "slow"
            fast_movement_threshold: Minimum movement percentage to be considered "fast"
            slow_duration_threshold: Minimum duration (seconds) to trigger slow walker roast
            roast_cooldown: Cooldown period (seconds) between roasts
            enable_voice: Whether to enable text-to-speech for roasts
            frame_skip: Process every Nth frame (higher = faster but less responsive)
            history_size: Number of frames to keep in movement history
        """
        self.camera_feed = camera_feed
        self.slow_movement_threshold = slow_movement_threshold
        self.fast_movement_threshold = fast_movement_threshold
        self.slow_duration_threshold = slow_duration_threshold
        self.roast_cooldown = roast_cooldown
        self.enable_voice = enable_voice
        self.frame_skip = frame_skip
        self.history_size = history_size
        
        # Initialize detection systems
        self.movement_detector = MovementDetection(
            camera_feed=camera_feed,
            movement_threshold=slow_movement_threshold,
            smoothing_frames=3  # Reduced for faster response
        )
        
        # Initialize ShameBot
        try:
            self.shame_bot = ShameBot()
            self.shame_bot_available = True
        except Exception as e:
            print(f"Warning: ShameBot not available: {e}")
            self.shame_bot_available = False
        
        # Slow walker detection state
        self.slow_walking_start_time = None
        self.last_roast_time = 0
        self.slow_walker_events = []
        self.is_slow_walking = False
        self.current_movement_speed = "normal"
        
        # Detection history for smoothing
        self.movement_history = []
        
        # Threading for non-blocking roasts
        self.roast_thread = None
        
        # Frame counter for skipping
        self.frame_counter = 0
        
        # Cached drawing elements for performance
        self._cached_overlay = None
        self._cached_stats = None
        self._last_stats_update = 0
        
    def analyze_walking_speed(self, movement_percentage: float) -> str:
        """
        Analyze movement percentage to determine walking speed category.
        
        Args:
            movement_percentage: Percentage of frame that is moving
            
        Returns:
            String indicating walking speed: 'slow', 'normal', 'fast', or 'none'
        """

        if movement_percentage < 0.003:
            return 'none'
        elif movement_percentage < self.slow_movement_threshold:
            return 'slow'
        elif movement_percentage > self.fast_movement_threshold:
            return 'fast'
        else:
            return 'normal'
    
    def detect_slow_walker(self, frame: np.ndarray) -> Dict:
        """
        Detect slow walker behavior in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with slow walker detection results
        """
        current_time = time.time()
        
        # Detect movement
        movement_detection = self.movement_detector.detect_movement(frame)
        movement_percentage = movement_detection['movement_percentage']
        contour_count = len(movement_detection['contours'])
        
        # Analyze walking speed
        walking_speed = self.analyze_walking_speed(movement_percentage)
        
        # Add to movement history
        self.movement_history.append({
            'timestamp': current_time,
            'movement_percentage': movement_percentage,
            'walking_speed': walking_speed,
            'contour_count': contour_count
        })
        
        # Keep only recent history
        if len(self.movement_history) > self.history_size:
            self.movement_history.pop(0)
        
        # Determine if slow walking is occurring with temporal smoothing
        slow_walking_detected = self._analyze_slow_walking_behavior()
        
        # Update slow walking state
        if slow_walking_detected and not self.is_slow_walking:
            # Slow walking just started
            self.slow_walking_start_time = current_time
            self.is_slow_walking = True
        elif not slow_walking_detected and self.is_slow_walking:
            # Slow walking stopped
            if self.slow_walking_start_time:
                duration = current_time - self.slow_walking_start_time
                if duration >= self.slow_duration_threshold:
                    # Record slow walker event
                    self.slow_walker_events.append({
                        'start_time': self.slow_walking_start_time,
                        'end_time': current_time,
                        'duration': duration,
                        'avg_movement': sum(d['movement_percentage'] for d in self.movement_history[-3:]) / 3
                    })
            self.is_slow_walking = False
            self.slow_walking_start_time = None
        
        # Check if we should trigger a roast
        roast_triggered = False
        if (self.is_slow_walking and 
            self.slow_walking_start_time and 
            current_time - self.slow_walking_start_time >= self.slow_duration_threshold and
            current_time - self.last_roast_time >= self.roast_cooldown):
            roast_triggered = True
            self.last_roast_time = current_time
            
            # Trigger roast in a separate thread to avoid blocking
            if self.shame_bot_available:
                self._trigger_roast_async(frame, movement_percentage)
                # save frame to file for reference
                cv2.imwrite("results/slow_walker_frame.jpg", frame)
        
        return {
            'slow_walking_detected': slow_walking_detected,
            'is_slow_walking': self.is_slow_walking,
            'roast_triggered': roast_triggered,
            'walking_speed': walking_speed,
            'movement_percentage': movement_percentage,
            'contour_count': contour_count,
            'movement_detection': movement_detection,
            'slow_walking_duration': current_time - self.slow_walking_start_time if self.slow_walking_start_time else 0,
            'total_slow_walker_events': len(self.slow_walker_events)
        }
    
    def _analyze_slow_walking_behavior(self) -> bool:
        """
        Analyze movement history to determine if slow walking is occurring.
        
        Returns:
            True if slow walking is detected, False otherwise
        """
        if len(self.movement_history) < 3:  # Reduced from 5 for faster response
            return False
        
        # Get recent detections
        recent_detections = self.movement_history[-3:]  # Reduced from 5
        
        # Count slow walking occurrences
        slow_count = sum(1 for d in recent_detections if d['walking_speed'] == 'slow')
        movement_count = sum(1 for d in recent_detections if d['movement_percentage'] > 0)
        
        # Require both slow movement and some movement for majority of recent frames
        slow_ratio = slow_count / len(recent_detections)
        movement_ratio = movement_count / len(recent_detections)
        
        # Reduced thresholds for faster response
        return slow_ratio >= 0.5 and movement_ratio >= 0.5  # Reduced from 0.6
    
    def _trigger_roast_async(self, frame: np.ndarray, movement_percentage: float):
        """
        Trigger a roast in a separate thread to avoid blocking the main detection loop.
        
        Args:
            frame: Current frame to roast
            movement_percentage: Current movement percentage
        """
        if self.roast_thread and self.roast_thread.is_alive():
            return  # Don't start multiple roast threads
        
        self.roast_thread = threading.Thread(
            target=self._generate_roast,
            args=(frame, movement_percentage)
        )
        self.roast_thread.daemon = True
        self.roast_thread.start()
    
    def _generate_roast(self, frame: np.ndarray, movement_percentage: float):
        """
        Generate and speak a roast for the slow walker.
        
        Args:
            frame: Current frame
            movement_percentage: Movement percentage
        """
        try:
            # Create information about the slow walker
            # info = f"Person walking very slowly with only {movement_percentage:.2%} movement detected"
            info = ""
            # Convert OpenCV frame to PIL Image for ShameBot
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Generate roast
            roast = self.shame_bot.create_roast(info, pil_image)
            print(f"🐌 SLOW WALKER ROAST: {roast}")
            
            # Speak the roast if voice is enabled
            if self.enable_voice:
                self.shame_bot.speak(roast)
                
        except Exception as e:
            print(f"Error generating roast: {e}")
    
    def get_slow_walker_statistics(self) -> Dict:
        """
        Get statistics about detected slow walker events.
        
        Returns:
            Dictionary with slow walker statistics
        """
        if not self.slow_walker_events:
            return {
                'total_events': 0,
                'average_duration': 0,
                'longest_duration': 0,
                'shortest_duration': 0,
                'average_movement': 0
            }
        
        durations = [event['duration'] for event in self.slow_walker_events]
        movements = [event['avg_movement'] for event in self.slow_walker_events]
        
        return {
            'total_events': len(self.slow_walker_events),
            'average_duration': sum(durations) / len(durations),
            'longest_duration': max(durations),
            'shortest_duration': min(durations),
            'average_movement': sum(movements) / len(movements),
            'recent_events': self.slow_walker_events[-5:]  # Last 5 events
        }
    
    def draw_slow_walker_detection(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw slow walker detection results on the frame.
        
        Args:
            frame: Input frame
            detection: Detection results from detect_slow_walker
            
        Returns:
            Frame with slow walker visualization
        """
        result_frame = frame.copy()
        
        # Draw movement detections (only if needed)
        if detection['is_slow_walking'] or detection['movement_percentage'] > 0.01:
            result_frame = self.movement_detector.draw_movement(
                result_frame, detection['movement_detection']
            )
        
        # Add slow walker status
        if detection['is_slow_walking']:
            # Use cached overlay for performance
            if self._cached_overlay is None or self._cached_overlay.shape != frame.shape:
                self._cached_overlay = np.zeros_like(frame)
                cv2.rectangle(self._cached_overlay, (0, 0), (frame.shape[1], 150), (0, 165, 255), -1)
            
            # Apply overlay
            result_frame = cv2.addWeighted(result_frame, 0.7, self._cached_overlay, 0.3, 0)
            
            # Slow walker alert text
            cv2.putText(result_frame, "SLOW WALKER DETECTED!", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Duration
            duration = detection['slow_walking_duration']
            cv2.putText(result_frame, f"Duration: {duration:.1f}s", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Movement percentage
            movement_pct = detection['movement_percentage']
            cv2.putText(result_frame, f"Movement: {movement_pct:.2%}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Roast indicator
            if detection['roast_triggered']:
                cv2.putText(result_frame, "ROASTING IN PROGRESS...", 
                           (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # Normal status - simplified for speed
            speed_colors = {
                'none': (128, 128, 128),
                'slow': (0, 165, 255),  # Orange
                'normal': (0, 255, 0),  # Green
                'fast': (0, 0, 255)     # Red
            }
            
            speed_color = speed_colors.get(detection['walking_speed'], (255, 255, 255))
            cv2.putText(result_frame, f"Speed: {detection['walking_speed'].upper()}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, speed_color, 2)
            
            cv2.putText(result_frame, f"Movement: {detection['movement_percentage']:.2%}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Cache statistics to avoid frequent recalculations
        current_time = time.time()
        if self._cached_stats is None or current_time - self._last_stats_update > 1.0:
            self._cached_stats = self.get_slow_walker_statistics()
            self._last_stats_update = current_time
        
        # Add statistics
        cv2.putText(result_frame, f"Events: {self._cached_stats['total_events']}", 
                   (10, result_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def run_detection_loop(self, display: bool = True, save_events: bool = True):
        """
        Run continuous slow walker detection.
        
        Args:
            display: Whether to display the video feed with detections
            save_events: Whether to save slow walker events to file
        """
        if not self.camera_feed.start():
            print("Failed to start camera")
            return
        
        print("Slow Walker Detection Started")
        print("Press 'q' to quit, 's' to show statistics, 'v' to toggle voice")
        
        try:
            while True:
                frame = self.camera_feed.get_frame()
                
                if frame is None:
                    print("Failed to get frame")
                    break
                
                # Frame skipping for performance
                self.frame_counter += 1
                if self.frame_counter % self.frame_skip != 0:
                    if display:
                        cv2.imshow('Slow Walker Detection', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                
                # Detect slow walker
                detection = self.detect_slow_walker(frame)
                
                # Print current status (less frequently for performance)
                if self.frame_counter % (self.frame_skip * 10) == 0:
                    status = "SLOW WALKING" if detection['is_slow_walking'] else "NORMAL"
                    print(f"\rStatus: {status} | Speed: {detection['walking_speed']} | Movement: {detection['movement_percentage']:.2%} | Events: {detection['total_slow_walker_events']}", 
                          end="", flush=True)
                
                # Handle roast trigger
                if detection['roast_triggered']:
                    print(f"\n🐌 SLOW WALKER ROAST TRIGGERED! Duration: {detection['slow_walking_duration']:.1f}s")
                    if save_events:
                        self._save_slow_walker_event(detection)
                
                if display:
                    # Draw detections on frame
                    result_frame = self.draw_slow_walker_detection(frame, detection)
                    
                    # Display frame
                    cv2.imshow('Slow Walker Detection', result_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._print_statistics()
                    elif key == ord('v'):
                        self.enable_voice = not self.enable_voice
                        print(f"\nVoice {'enabled' if self.enable_voice else 'disabled'}")
                
                # Reduced delay for higher FPS
                time.sleep(0.01)  # ~100 FPS when frame_skip=2
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.camera_feed.stop()
            if display:
                cv2.destroyAllWindows()
            print("\nSlow Walker Detection Stopped")
            
            # Print final statistics
            self._print_statistics()
    
    def _save_slow_walker_event(self, detection: Dict):
        """
        Save slow walker event to file.
        
        Args:
            detection: Detection results containing slow walker event
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        duration = detection['slow_walking_duration']
        movement_pct = detection['movement_percentage']
        
        with open("slow_walker_events.log", "a") as f:
            f.write(f"{timestamp} - Slow walker detected for {duration:.1f}s (movement: {movement_pct:.2%})\n")
    
    def _print_statistics(self):
        """
        Print slow walker detection statistics.
        """
        stats = self.get_slow_walker_statistics()
        print(f"\n=== Slow Walker Detection Statistics ===")
        print(f"Total Events: {stats['total_events']}")
        if stats['total_events'] > 0:
            print(f"Average Duration: {stats['average_duration']:.1f} seconds")
            print(f"Longest Duration: {stats['longest_duration']:.1f} seconds")
            print(f"Shortest Duration: {stats['shortest_duration']:.1f} seconds")
            print(f"Average Movement: {stats['average_movement']:.2%}")
            print(f"Recent Events: {len(stats['recent_events'])}")
        print("=========================================")


def main():
    """
    Example usage of the SlowWalkerDetection class.
    """
    # Create camera feed
    camera = CameraFeed()
    
    # Create slow walker detector with optimized settings
    detector = SlowWalkerDetection(
        camera,
        frame_skip=2,  # Process every 2nd frame for speed
        history_size=5,  # Reduced history for faster response
        slow_duration_threshold=0.8  # Slightly reduced for faster detection
    )
    
    # Run detection
    detector.run_detection_loop()


if __name__ == "__main__":
    main() 