#!/usr/bin/env python3
"""
Test script to measure the performance improvements of the optimized slow walker detection.
"""

import time
import cv2
import numpy as np
from camera.camera_feed import CameraFeed
from models.slow_walker_detection import SlowWalkerDetection


def test_performance():
    """Test the performance of the slow walker detection system."""
    
    print("🚀 Testing Slow Walker Detection Performance")
    print("=" * 50)
    
    # Create camera feed
    camera = CameraFeed()
    
    # Test different configurations
    configs = [
        {
            'name': 'Optimized (Fast)',
            'frame_skip': 3,
            'history_size': 3,
            'slow_duration_threshold': 0.5
        },
        {
            'name': 'Balanced',
            'frame_skip': 2,
            'history_size': 5,
            'slow_duration_threshold': 0.8
        },
        {
            'name': 'Sensitive',
            'frame_skip': 1,
            'history_size': 7,
            'slow_duration_threshold': 1.0
        }
    ]
    
    for config in configs:
        print(f"\n📊 Testing {config['name']} Configuration:")
        print(f"   Frame Skip: {config['frame_skip']}")
        print(f"   History Size: {config['history_size']}")
        print(f"   Duration Threshold: {config['slow_duration_threshold']}s")
        
        # Create detector with current config
        detector = SlowWalkerDetection(
            camera,
            frame_skip=config['frame_skip'],
            history_size=config['history_size'],
            slow_duration_threshold=config['slow_duration_threshold']
        )
        
        # Start camera
        if not camera.start():
            print("   ❌ Failed to start camera")
            continue
        
        # Performance test
        frame_count = 0
        total_time = 0
        detection_times = []
        
        print("   🔄 Running performance test (10 seconds)...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < 10:  # Test for 10 seconds
                frame = camera.get_frame()
                if frame is None:
                    continue
                
                # Measure detection time
                detection_start = time.time()
                detection = detector.detect_slow_walker(frame)
                detection_time = time.time() - detection_start
                
                detection_times.append(detection_time)
                frame_count += 1
                total_time += detection_time
                
                # Display frame without drawing for pure performance test
                cv2.imshow('Performance Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("   ⏹️ Test interrupted")
        
        finally:
            camera.stop()
            cv2.destroyAllWindows()
        
        # Calculate statistics
        if frame_count > 0:
            avg_time = total_time / frame_count
            avg_fps = frame_count / (time.time() - start_time)
            min_time = min(detection_times)
            max_time = max(detection_times)
            
            print(f"   ✅ Results:")
            print(f"      Frames processed: {frame_count}")
            print(f"      Average FPS: {avg_fps:.1f}")
            print(f"      Average detection time: {avg_time*1000:.1f}ms")
            print(f"      Min detection time: {min_time*1000:.1f}ms")
            print(f"      Max detection time: {max_time*1000:.1f}ms")
            print(f"      Total events detected: {detection.get('total_slow_walker_events', 0)}")
        else:
            print("   ❌ No frames processed")


def test_movement_detection_speed():
    """Test the speed of movement detection specifically."""
    
    print("\n🔍 Testing Movement Detection Speed")
    print("=" * 40)
    
    from models.movement_detection import MovementDetection
    
    camera = CameraFeed()
    movement_detector = MovementDetection(camera)
    
    if not camera.start():
        print("❌ Failed to start camera")
        return
    
    # Test both methods
    methods = ['background', 'frame_diff']
    
    for method in methods:
        print(f"\n📈 Testing {method} method:")
        
        frame_count = 0
        total_time = 0
        detection_times = []
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < 5:  # Test for 5 seconds
                frame = camera.get_frame()
                if frame is None:
                    continue
                
                # Measure detection time
                detection_start = time.time()
                detection = movement_detector.detect_movement(frame, method)
                detection_time = time.time() - detection_start
                
                detection_times.append(detection_time)
                frame_count += 1
                total_time += detection_time
                
        except KeyboardInterrupt:
            break
        
        finally:
            pass  # Don't stop camera yet
        
        # Calculate statistics
        if frame_count > 0:
            avg_time = total_time / frame_count
            avg_fps = frame_count / (time.time() - start_time)
            
            print(f"   Frames: {frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Average time: {avg_time*1000:.1f}ms")
            print(f"   Movement detected: {detection['movement_percentage']:.2%}")
    
    camera.stop()


def test_drawing_performance():
    """Test the performance impact of drawing operations."""
    
    print("\n🎨 Testing Drawing Performance")
    print("=" * 35)
    
    camera = CameraFeed()
    detector = SlowWalkerDetection(camera, frame_skip=1)
    
    if not camera.start():
        print("❌ Failed to start camera")
        return
    
    # Test with and without drawing
    tests = [
        {'name': 'Without Drawing', 'draw': False},
        {'name': 'With Drawing', 'draw': True}
    ]
    
    for test in tests:
        print(f"\n📊 Testing {test['name']}:")
        
        frame_count = 0
        total_time = 0
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < 5:  # Test for 5 seconds
                frame = camera.get_frame()
                if frame is None:
                    continue
                
                # Measure processing time
                process_start = time.time()
                detection = detector.detect_slow_walker(frame)
                
                if test['draw']:
                    result_frame = detector.draw_slow_walker_detection(frame, detection)
                    cv2.imshow('Drawing Test', result_frame)
                else:
                    cv2.imshow('Drawing Test', frame)
                
                process_time = time.time() - process_start
                
                frame_count += 1
                total_time += process_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            break
        
        finally:
            pass
        
        # Calculate statistics
        if frame_count > 0:
            avg_time = total_time / frame_count
            avg_fps = frame_count / (time.time() - start_time)
            
            print(f"   Frames: {frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Average time: {avg_time*1000:.1f}ms")
    
    camera.stop()
    cv2.destroyAllWindows()


def main():
    """Run all performance tests."""
    
    print("🏃‍♂️ Slow Walker Detection Performance Test Suite")
    print("=" * 55)
    
    try:
        # Test overall performance with different configs
        test_performance()
        
        # Test movement detection speed
        test_movement_detection_speed()
        
        # Test drawing performance
        test_drawing_performance()
        
        print("\n✅ All performance tests completed!")
        print("\n💡 Performance Tips:")
        print("   - Use frame_skip=2 or 3 for better performance")
        print("   - Reduce history_size for faster response")
        print("   - Disable drawing for pure detection speed")
        print("   - Use background subtraction method for better accuracy")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    main() 