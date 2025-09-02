#!/usr/bin/env python3
"""
Test data pulling functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from option_pricing.data.yahoo_data import OptionDataPuller

print("Testing PLTR data pull...")

try:
    puller = OptionDataPuller("PLTR")
    
    # Get current price
    print("Getting current price...")
    price = puller.get_current_price()
    print(f"Current PLTR Price: ${price:.2f}")
    
    # Get expirations
    print("\nGetting option expirations...")
    expirations = puller.get_option_expirations()
    print(f"Found {len(expirations)} expirations")
    
    if expirations:
        print(f"Next expiration: {expirations[0]}")
        
        # Get one option chain
        print("\nGetting option chain for nearest expiration...")
        chain = puller.get_option_chain(expirations[0])
        
        if not chain['calls'].empty:
            print(f"Found {len(chain['calls'])} call options")
            print(f"Found {len(chain['puts'])} put options")
        else:
            print("No options found")
    
    print("\n✓ Data pull test successful!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()