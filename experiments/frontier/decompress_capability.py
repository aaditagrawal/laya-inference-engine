"""Query dedicated hardware decompression support without allocating GPU memory."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from cuda.bindings import driver as cuda


def checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(str(status))
    return values[0] if len(values) == 1 else values


def main():
    checked(cuda.cuInit(0))
    device = checked(cuda.cuDeviceGet(0))
    name = checked(cuda.cuDeviceGetName(256, device))
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "device": name.split(b"\0", 1)[0].decode(),
        "driver_version": checked(cuda.cuDriverGetVersion()),
        "scope": "Read-only device attributes. No allocation, kernel, benchmark, or software-decompression fallback.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_documentation": "https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html",
        "attributes": {},
    }
    for attr_name in [
        "CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK",
        "CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH",
    ]:
        result = cuda.cuDeviceGetAttribute(
            getattr(cuda.CUdevice_attribute, attr_name), device
        )
        report["attributes"][attr_name] = {
            "status": str(result[0]),
            "status_code": int(result[0]),
            "value": int(result[1]) if int(result[0]) == 0 else None,
        }
    mask = report["attributes"]["CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK"]
    report["hardware_formats_available"] = mask["status_code"] == 0 and bool(
        mask["value"]
    )
    Path("results/frontier/decompress-capability.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
