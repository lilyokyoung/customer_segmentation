#!/usr/bin/env python3
"""
Simple entry point for deployment platforms
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Import and run the main function
    try:
        from main import main
        import asyncio
        asyncio.run(main())
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)
