#!/usr/bin/env python3
"""
Fire Detection System - One Command Runner
Run this script to start the fire detection web application.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, socketio

if __name__ == '__main__':
    print("🔥 Starting Fire Detection System...")
    print("📱 Web interface will be available at: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        socketio.run(app, debug=True, port=5001, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n🛑 Fire Detection System stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
