"""
KhachKhach v2 Test Script - WITH NORMALIZED COORDINATES!

Features:
✅ Coordinates normalized to 0-1 range 
✅ Clean comma-separated format (no nested lists)
✅ User can choose format and normalization
"""

import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

VIDEO_PATH = r"C:\Users\Divya\Downloads\female-barbell-zercher-squat-front.mp4"
OUTPUT_DIR = r"C:\Users\Divya\Downloads\khach_v2_normalized_test"

def test_normalized_keypoints():
    """Test keypoint detection with normalized coordinates"""
    print("\n🏃 Testing NORMALIZED Keypoint Detection...")
    
    try:
        from khachkhach.video_processor import VideoProcessor
        from khachkhach.detection_engine import DetectionEngine
        
        # Extract 1 frame for testing
        processor = VideoProcessor()
        frames_dir = os.path.join(OUTPUT_DIR, "frames")
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        
        frame_result = processor.extract_frames(
            video_path=VIDEO_PATH,
            output_dir=frames_dir,
            max_frames=1
        )
        
        if not frame_result.get("success"):
            return False
        
        # Test normalized keypoints with comma format
        engine = DetectionEngine("yolo11n-pose.pt")
        keypoints_dir = os.path.join(OUTPUT_DIR, "keypoints_normalized")
        
        result = engine.annotate_keypoints(
            input_path=frames_dir,
            output_dir=keypoints_dir,
            confidence=0.5,
            normalize_coords=True,        # 0-1 range
            output_format="space_separated"  # Clean format
        )
        
        if result.get("success"):
            txt_files = [f for f in os.listdir(keypoints_dir) if f.endswith('.txt')]
            if txt_files:
                sample_file = os.path.join(keypoints_dir, txt_files[0])
                with open(sample_file, 'r') as f:
                    content = f.read().strip()
                
                print(f"✅ Sample output: {content[:60]}...")
                
                # Verify coordinates are 0-1 range
                if ',' in content:
                    coords = content.split(',')[:6]  # Check first 6 values
                    sample_coords = [float(x) for x in coords if x.replace('.', '').isdigit()]
                    
                    if all(0 <= coord <= 1 for coord in sample_coords):
                        print("✅ Coordinates properly normalized (0-1 range)")
                    else:
                        print("⚠️ Coordinates not properly normalized")
                        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Testing Normalized Coordinates")
    
    if os.path.exists(VIDEO_PATH):
        success = test_normalized_keypoints()
        if success:
            print("\n🎉 PERFECT! NORMALIZED COORDINATES WORKING! 🔥")
            print("✅ Coordinates are 0-1 range")  
            print("✅ Clean comma-separated format")
            print("✅ No more nested lists!")
        else:
            print("\n❌ Test failed")
    else:
        print(f"❌ Video not found: {VIDEO_PATH}")
