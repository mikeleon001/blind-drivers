# evaluation/

Metrics, logging, and test scripts used to assess system performance.

Metrics tracked:
- Lateral deviation from lane center (mean, max, RMSE)
- Correction responsiveness (time to return to lane center after a deviation)
- Lane detection accuracy

Tests are run across multiple CARLA maps, road geometries (straight/curved), and
weather/lighting conditions.
