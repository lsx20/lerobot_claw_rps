import sys
from pathlib import Path


repo_vendor_sdk = Path(__file__).resolve().parents[2] / "vendor" / "piper_sdk"
if repo_vendor_sdk.exists():
    vendor_path = str(repo_vendor_sdk)
    if vendor_path in sys.path:
        sys.path.remove(vendor_path)
    sys.path.insert(0, vendor_path)
