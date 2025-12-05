import numpy as np
import cv2
from scipy.ndimage import map_coordinates

class Probe:
    def __init__(self, probe_id, probe_type, coords, color, label=None):
        self.id = probe_id
        self.type = probe_type  # 'point', 'line', 'area'
        self.coords = coords    # (x, y) for point
        self.color = color
        self.label = label if label else f"P{probe_id}"

class ProbeManager:
    def __init__(self):
        self.probes = []
        self._next_point_id = 1
        self._next_line_id = 1
        self._next_area_id = 1
        # Distinct colors for probes
        self.colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']

    def add_point(self, x, y):
        """Add a new point probe."""
        color = self.colors[(self._next_point_id - 1) % len(self.colors)]
        probe = Probe(self._next_point_id, 'point', (x, y), color)
        self.probes.append(probe)
        self._next_point_id += 1
        return probe

    def remove_probe(self, probe_id):
        """Remove a probe by ID."""
        # Note: IDs might not be unique across types now, so we should filter by type too if needed.
        # But for now, let's assume we pass the object or handle it in UI.
        # Actually, remove_probe(id) is ambiguous if IDs are reused.
        # Better to remove by (id, type) or just unique ID.
        # Given the UI structure, we select from a specific list (Point List or Line List).
        # So we should probably add a type argument to remove_probe.
        pass 

    def remove_probe_by_id_type(self, probe_id, probe_type):
        self.probes = [p for p in self.probes if not (p.id == probe_id and p.type == probe_type)]

    def clear_all(self):
        """Remove all probes."""
        self.probes = []
        self._next_point_id = 1
        self._next_line_id = 1
        self._next_area_id = 1

    def add_line(self, p1, p2):
        """Add a new line probe."""
        color = self.colors[(self._next_line_id - 1) % len(self.colors)]
        # coords for line is [p1, p2] where p1=(x1, y1)
        probe = Probe(self._next_line_id, 'line', [p1, p2], color)
        self.probes.append(probe)
        self._next_line_id += 1
        return probe

    def extract_time_series(self, data_list, scale_factors=(1.0, 1.0), offset=(0, 0)):
        """
        Extract time series data for all POINT probes.
        """
        results = {p.id: [] for p in self.probes if p.type == 'point'}
        
        if not data_list:
            return results

        # Pre-calculate coordinates for map_coordinates
        sy, sx = scale_factors
        off_x, off_y = offset
        
        point_coords = []
        point_ids = []
        
        for p in self.probes:
            if p.type == 'point':
                # Apply offset then scale
                px = (p.coords[0] - off_x) * sx
                py = (p.coords[1] - off_y) * sy
                point_coords.append([py, px]) # y, x
                point_ids.append(p.id)
                
        if not point_coords:
            return results
            
        point_coords = np.array(point_coords).T # Shape (2, N_probes)
        
        for frame_data in data_list:
            if frame_data is None:
                for pid in point_ids:
                    results[pid].append(np.nan)
                continue
                
            values = map_coordinates(frame_data, point_coords, order=1, mode='constant', cval=np.nan)
            
            for i, pid in enumerate(point_ids):
                results[pid].append(values[i])
                
        return results

import numpy as np
from scipy.ndimage import map_coordinates

class Probe:
    def __init__(self, probe_id, probe_type, coords, color, label=None):
        self.id = probe_id
        self.type = probe_type  # 'point', 'line', 'area'
        self.coords = coords    # (x, y) for point
        self.color = color
        self.label = label if label else f"P{probe_id}"

class ProbeManager:
    def __init__(self):
        self.probes = []
        self._next_point_id = 1
        self._next_line_id = 1
        self._next_area_id = 1
        # Distinct colors for probes
        self.colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']

    def add_point(self, x, y):
        """Add a new point probe."""
        color = self.colors[(self._next_point_id - 1) % len(self.colors)]
        probe = Probe(self._next_point_id, 'point', (x, y), color)
        self.probes.append(probe)
        self._next_point_id += 1
        return probe

    def remove_probe(self, probe_id):
        """Remove a probe by ID."""
        # Note: IDs might not be unique across types now, so we should filter by type too if needed.
        # But for now, let's assume we pass the object or handle it in UI.
        # Actually, remove_probe(id) is ambiguous if IDs are reused.
        # Better to remove by (id, type) or just unique ID.
        # Given the UI structure, we select from a specific list (Point List or Line List).
        # So we should probably add a type argument to remove_probe.
        pass 

    def remove_probe_by_id_type(self, probe_id, probe_type):
        self.probes = [p for p in self.probes if not (p.id == probe_id and p.type == probe_type)]

    def clear_all(self):
        """Remove all probes."""
        self.probes = []
        self._next_point_id = 1
        self._next_line_id = 1
        self._next_area_id = 1

    def clear_by_type(self, probe_type):
        """Remove all probes of a specific type."""
        self.probes = [p for p in self.probes if p.type != probe_type]
        # Reset ID counter for that type
        if probe_type == 'point':
            self._next_point_id = 1
        elif probe_type == 'line':
            self._next_line_id = 1
        elif probe_type == 'area':
            self._next_area_id = 1

    def add_line(self, p1, p2):
        """Add a new line probe."""
        color = self.colors[(self._next_line_id - 1) % len(self.colors)]
        # coords for line is [p1, p2] where p1=(x1, y1)
        probe = Probe(self._next_line_id, 'line', [p1, p2], color)
        self.probes.append(probe)
        self._next_line_id += 1
        return probe

    def extract_time_series(self, data_list, scale_factors=(1.0, 1.0), offset=(0, 0)):
        """
        Extract time series data for all POINT probes.
        """
        results = {p.id: [] for p in self.probes if p.type == 'point'}
        
        if not data_list:
            return results

        # Pre-calculate coordinates for map_coordinates
        sy, sx = scale_factors
        off_x, off_y = offset
        
        point_coords = []
        point_ids = []
        
        for p in self.probes:
            if p.type == 'point':
                # Apply offset then scale
                px = (p.coords[0] - off_x) * sx
                py = (p.coords[1] - off_y) * sy
                point_coords.append([py, px]) # y, x
                point_ids.append(p.id)
                
        if not point_coords:
            return results
            
        point_coords = np.array(point_coords).T # Shape (2, N_probes)
        
        for frame_data in data_list:
            if frame_data is None:
                for pid in point_ids:
                    results[pid].append(np.nan)
                continue
                
            values = map_coordinates(frame_data, point_coords, order=1, mode='constant', cval=np.nan)
            
            for i, pid in enumerate(point_ids):
                results[pid].append(values[i])
                
        return results

    def extract_kymograph(self, data_list, line_id, num_points=100, scale_factors=(1.0, 1.0), offset=(0, 0)):
        """
        Extract Kymograph data for a specific line probe.
        Returns:
            kymograph: 2D array (num_points, num_frames)
            dist_axis: 1D array of distances along the line
        """
        # Find the line probe
        line_probe = next((p for p in self.probes if p.id == line_id and p.type == 'line'), None)
        if not line_probe:
            return None, None
            
        p1, p2 = line_probe.coords
        
        # Generate sample points along the line
        x_samples = np.linspace(p1[0], p2[0], num_points)
        y_samples = np.linspace(p1[1], p2[1], num_points)
        
        # Calculate distance axis
        dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        dist_axis = np.linspace(0, dist, num_points)
        
        # Apply scaling and offset to map to data coordinates
        sy, sx = scale_factors
        off_x, off_y = offset
        
        # Map to data coordinates: (x - off_x) * sx, (y - off_y) * sy
        # Note: map_coordinates expects (row, col) -> (y, x)
        sample_coords_y = (y_samples - off_y) * sy
        sample_coords_x = (x_samples - off_x) * sx
        
        num_frames = len(data_list)
        kymograph = np.zeros((num_points, num_frames))
        
        for i, frame_data in enumerate(data_list):
            if frame_data is None:
                kymograph[:, i] = np.nan
                continue
                
            # Interpolate
            # map_coordinates uses (row, col) order, i.e., (y, x)
            vals = map_coordinates(frame_data, [sample_coords_y, sample_coords_x], order=1, mode='nearest')
            kymograph[:, i] = vals
            
        return kymograph, dist_axis

    def extract_line_series(self, data_list, line_id, metric='avg', num_points=100, scale_factors=(1.0, 1.0), offset=(0, 0)):
        """
        Extract time series data for a LINE probe based on a metric (avg, max, min).
        """
        kymograph, _ = self.extract_kymograph(data_list, line_id, num_points, scale_factors, offset)
        
        if kymograph is None:
            return None
            
        # kymograph shape: (num_points, num_frames)
        # We want to reduce along axis 0 (points) to get (num_frames,)
        
        if metric == 'max':
            return np.nanmax(kymograph, axis=0)
        elif metric == 'min':
            return np.nanmin(kymograph, axis=0)
        else: # avg
            return np.nanmean(kymograph, axis=0)

    def add_area(self, shape_type, coords):
        """Add a new area probe."""
        color = self.colors[(self._next_area_id - 1) % len(self.colors)]
        # coords depends on shape:
        # rect: [x0, y0, x1, y1]
        # circle: [cx, cy, radius]
        # poly: [[x,y], [x,y], ...]
        probe = Probe(self._next_area_id, 'area', {'shape': shape_type, 'data': coords}, color)
        self.probes.append(probe)
        self._next_area_id += 1
        return probe

    def extract_area_series(self, data_list, area_id, metric='avg', scale_factors=(1.0, 1.0), offset=(0, 0)):
        """
        Extract time series data for an AREA probe.
        """
        area_probe = next((p for p in self.probes if p.id == area_id and p.type == 'area'), None)
        if not area_probe:
            return None
            
        if not data_list or data_list[0] is None:
            return None
            
        # Create mask for the shape
        h, w = data_list[0].shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        sy, sx = scale_factors
        off_x, off_y = offset
        
        shape_type = area_probe.coords['shape']
        shape_data = area_probe.coords['data']
        
        # Helper to transform coords to data space
        def transform_point(x, y):
            return int((x - off_x) * sx), int((y - off_y) * sy)
            
        if shape_type == 'rect':
            x0, y0, x1, y1 = shape_data
            tx0, ty0 = transform_point(x0, y0)
            tx1, ty1 = transform_point(x1, y1)
            # Ensure correct order
            tx0, tx1 = min(tx0, tx1), max(tx0, tx1)
            ty0, ty1 = min(ty0, ty1), max(ty0, ty1)
            cv2.rectangle(mask, (tx0, ty0), (tx1, ty1), 1, -1)
            
        elif shape_type == 'circle':
            cx, cy, r = shape_data
            tcx, tcy = transform_point(cx, cy)
            # Scale radius (approximate with mean scale if non-uniform)
            tr = int(r * (sx + sy) / 2)
            cv2.circle(mask, (tcx, tcy), tr, 1, -1)
            
        elif shape_type == 'poly':
            pts = []
            for px, py in shape_data:
                pts.append(transform_point(px, py))
            pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, pts, 1)
            
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return [np.nan] * len(data_list)
            
        results = []
        for frame_data in data_list:
            if frame_data is None:
                results.append(np.nan)
                continue
                
            # Extract masked values
            masked_vals = frame_data[mask_bool]
            
            if masked_vals.size == 0:
                results.append(np.nan)
                continue
                
            if metric == 'max':
                results.append(np.nanmax(masked_vals))
            elif metric == 'min':
                results.append(np.nanmin(masked_vals))
            else: # avg
                results.append(np.nanmean(masked_vals))
                
        return results

    def to_list(self):
        # Convert numpy types to python types for JSON serialization if needed
        # But here we just return dicts
        return [p.to_dict() for p in self.probes]

    def load_from_list(self, data_list):
        self.probes = [Probe.from_dict(d) for d in data_list]
        # Reset counters
        self._next_point_id = max([p.id for p in self.probes if p.type == 'point'] + [0]) + 1
        self._next_line_id = max([p.id for p in self.probes if p.type == 'line'] + [0]) + 1
        self._next_area_id = max([p.id for p in self.probes if p.type == 'area'] + [0]) + 1
