#!/usr/bin/env python3
"""Run the CloudXR service from an installed CloudXR SDK."""

import ctypes
import os
from pathlib import Path
import signal
import sys


lib_path = Path(os.environ.get("CXR_LIB_PATH", ""))
if not lib_path.is_file():
    sys.exit("Set CXR_LIB_PATH to the CloudXR SDK libcloudxr.so")

runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
for name in ("runtime_started", "cloudxr.pid", "ipc_cloudxr"):
    (runtime_dir / name).unlink(missing_ok=True)

lib = ctypes.CDLL(str(lib_path))
service = ctypes.c_void_p()
result = lib.nv_cxr_service_create(ctypes.byref(service))
if result:
    sys.exit(f"CloudXR service creation failed: {result}")


def stop(_signum, _frame):
    lib.nv_cxr_service_stop(service)


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

result = lib.nv_cxr_service_start(service)
if result:
    lib.nv_cxr_service_destroy(service)
    sys.exit(f"CloudXR service start failed: {result}")

print("CloudXR service ready on port 49100", flush=True)
lib.nv_cxr_service_join(service)
lib.nv_cxr_service_destroy(service)
