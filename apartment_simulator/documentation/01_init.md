# Backend Module: `__init__.py`

**File Path:** `apartment_simulator/backend/__init__.py`  
**Purpose:** Package initialization and module declaration  
**Status:** Minimal initialization (standard Python package pattern)

---

## Overview

The `__init__.py` file marks the `backend` directory as a Python package, enabling it to be imported as a module from parent directories and other packages.

## What It Does

```python
# Backend package initialization
```

This simple one-line comment serves as:
1. **Package Marker**: Tells Python that `backend/` is a package (importable module)
2. **Documentation**: Briefly describes the package purpose
3. **Empty Interface**: Currently exports no symbols directly from the package

## How It Works

### Python Package Structure

When Python encounters a directory with `__init__.py`, it treats the directory as a package:

```
apartment_simulator/
├── backend/
│   ├── __init__.py          ← Makes backend a package
│   ├── live_simulator.py
│   ├── model.py
│   └── server.py
├── frontend/
└── preprocessing/
```

### Import Mechanisms

With this file in place, the following imports become possible:

```python
# From within apartment_simulator/
from backend import live_simulator
from backend.live_simulator import LiveWaterFlowGenerator
from backend.model import HybridWaterAnomalyDetector

# Full path imports
from apartment_simulator.backend.live_simulator import LiveApartmentBuildingDataGenerator
from apartment_simulator.backend.server import app, sio
```

## Current Configuration

### Content
```python
# Backend package initialization
```

### What's Not Configured (Currently Empty)

The `__init__.py` does **not** currently:
- ✗ Import and re-export classes (not in `__all__`)
- ✗ Define module-level constants
- ✗ Initialize shared resources
- ✗ Set up logging or configuration

### Why This Minimal Approach?

**Pros:**
- Clean separation of concerns
- Explicit imports required (clarity in code)
- Minimal coupling between modules
- Easy to add configuration later

**Trade-offs:**
- Users must specify full import paths
- No convenient one-liner imports like `from backend import Generator, Detector`

## Usage Examples

### Direct Imports (Current Pattern)
```python
# In server.py or other frontend code
from backend.live_simulator import LiveApartmentBuildingDataGenerator
from backend.model import HybridWaterAnomalyDetector

generator = LiveApartmentBuildingDataGenerator(...)
detector = HybridWaterAnomalyDetector(...)
```

### Potential Enhanced Pattern (Future)

If we wanted to provide convenience imports, we could modify `__init__.py` to:

```python
"""
Backend package for apartment simulator.

Provides:
- Real-time water flow generation for 50-apartment buildings
- Hybrid anomaly detection (CUSUM + Isolation Forest)
- FastAPI server with WebSocket streaming
"""

from .live_simulator import (
    LiveWaterFlowGenerator,
    LiveApartmentBuildingDataGenerator,
)
from .model import HybridWaterAnomalyDetector
from .server import app, sio

__all__ = [
    'LiveWaterFlowGenerator',
    'LiveApartmentBuildingDataGenerator',
    'HybridWaterAnomalyDetector',
    'app',
    'sio',
]
```

**This would enable:**
```python
from backend import LiveApartmentBuildingDataGenerator, HybridWaterAnomalyDetector
```

## File Structure Summary

| Aspect | Details |
|--------|---------|
| **File Name** | `__init__.py` |
| **Location** | `apartment_simulator/backend/` |
| **Size** | 1 line |
| **Dependencies** | None (standard Python) |
| **Exports** | None (currently) |
| **Purpose** | Package marker |

## Modification Guide

### To Add Convenience Imports

Replace the current content with:

```python
"""Backend package - Real-time water flow simulation and anomaly detection."""

from .live_simulator import LiveWaterFlowGenerator, LiveApartmentBuildingDataGenerator
from .model import HybridWaterAnomalyDetector

__all__ = ['LiveWaterFlowGenerator', 'LiveApartmentBuildingDataGenerator', 'HybridWaterAnomalyDetector']
```

### To Add Configuration

```python
"""Backend package configuration and initialization."""

import logging

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Version info
__version__ = "1.0.0"
__author__ = "Water Detection System"

# Import main classes
from .live_simulator import LiveWaterFlowGenerator, LiveApartmentBuildingDataGenerator
from .model import HybridWaterAnomalyDetector

__all__ = [
    'LiveWaterFlowGenerator',
    'LiveApartmentBuildingDataGenerator', 
    'HybridWaterAnomalyDetector',
    '__version__',
    '__author__',
    'logger',
]
```

## Integration with Other Modules

### Referenced By

- **`server.py`**: Imports from this package
  ```python
  from live_simulator import LiveApartmentBuildingDataGenerator
  from model import HybridWaterAnomalyDetector
  ```

- **Frontend/External Code**: Uses package for data generation and detection

## Best Practices

1. **Minimal Dependencies**: Keep `__init__.py` lightweight to avoid circular imports
2. **Explicit Exports**: Use `__all__` if providing convenience imports
3. **Documentation**: Add docstring explaining package purpose
4. **Version Control**: Keep version synchronized with releases
5. **Logging**: Use module-level logger for diagnostics

## Summary

| Component | Status |
|-----------|--------|
| **Current State** | Minimal initialization |
| **Purpose** | Package marker |
| **Complexity** | Very Low |
| **Configuration** | None |
| **Exports** | None |
| **Enhancement Potential** | High (can add convenience imports, config) |

---

**End of `__init__.py` Documentation**
