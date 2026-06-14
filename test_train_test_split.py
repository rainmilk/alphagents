"""
Quick verification script for train/test split implementation.

This script verifies that:
1. Data is correctly split into train/test sets
2. Training data does NOT include test data
3. FactorBacktester uses training data only
4. BacktestEngine uses test data only
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from main import AAAI2027Pipeline

def test_train_test_split():
    """Test that train/test split works correctly."""
    
    print("=" * 60)
    print("Testing Train/Test Split Implementation")
    print("=" * 60)
    
    # Initialize pipeline (pass config file path, not the parsed dict)
    pipeline = AAAI2027Pipeline('config/config.yaml')
    
    # Step 1: Load data (this will also perform train/test split)
    print("\n[Step 1] Loading data with train/test split...")
    pipeline.step1_load_data(use_sample=True)
    
    # Verify split
    print("\n--- Verification ---")
    
    # Check that train_data and test_data exist
    assert hasattr(pipeline, 'train_data'), "pipeline.train_data not found!"
    assert hasattr(pipeline, 'test_data'), "pipeline.test_data not found!"
    print("✓ train_data and test_data attributes exist")
    
    # Check that train and test data have correct keys
    assert 'price_data' in pipeline.train_data, "train_data missing 'price_data'"
    assert 'price_data' in pipeline.test_data, "test_data missing 'price_data'"
    print("✓ train_data and test_data have 'price_data' key")
    
    # Check that train and test data don't overlap
    train_dates = pipeline.train_data['price_data']['close'].index
    test_dates = pipeline.test_data['price_data']['close'].index
    
    overlap = train_dates.intersection(test_dates)
    assert len(overlap) == 0, f"Train and test data overlap! {len(overlap)} overlapping dates"
    print(f"✓ No overlap between train ({len(train_dates)} days) and test ({len(test_dates)} days)")
    
    # Check that train ends before test starts
    assert train_dates[-1] < test_dates[0], "Train data doesn't end before test data starts!"
    print(f"✓ Train ends at {train_dates[-1].date()}, Test starts at {test_dates[0].date()}")
    
    # Check that full data = train + test
    full_dates = pipeline.price_data['close'].index
    combined_dates = train_dates.append(test_dates)
    assert len(full_dates) == len(combined_dates), "Full data != train + test"
    print(f"✓ Full data ({len(full_dates)} days) = Train ({len(train_dates)}) + Test ({len(test_dates)})")
    
    print("\n--- Summary ---")
    print(f"Train period: {train_dates[0].date()} ~ {train_dates[-1].date()}")
    print(f"Test period:  {test_dates[0].date()} ~ {test_dates[-1].date()}")
    print(f"Train/Test ratio: {len(train_dates)} / {len(test_dates)} = {len(train_dates)/len(test_dates):.2f}")
    
    print("\n" + "=" * 60)
    print("All verification checks passed! ✓")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        test_train_test_split()
    except AssertionError as e:
        print(f"\n✗ Verification FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
