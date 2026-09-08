"""
Class for loading data from hierarchical HDF5 files with ring-based structure
"""

import h5py
import numpy as np
from pathlib import Path
from glob import glob
from torch.utils.data import Dataset
from abc import ABC


class HierarchicalH5Dataset(Dataset, ABC):
    """
    Dataset class for loading ring-based data from hierarchical HDF5 files.
    
    Each HDF5 file contains multiple events, and each event contains multiple rings.
    Each ring is treated as a separate sample.
    
    File structure:
    /event_XXXXXX/ring_Y/energy, tube_ids, pmt_charge, pmt_time, etc.
    
    Attributes loaded per ring:
    - energy: float (SCALAR)
    - event_type: int (SCALAR)
    - particle_dir_x, particle_dir_y, particle_dir_z: float (SCALAR)
    - particle_start_x, particle_start_y, particle_start_z: float (SCALAR)
    - tube_ids: array of int (variable length)
    - pmt_charge: array of float (variable length)
    - pmt_time: array of float (variable length)
    """
    
    def __init__(self, file_pattern, pmt_positions_file, use_memmap=False, one_indexed=False):
        """
        Initialize the hierarchical HDF5 dataset.
        
        Parameters
        ----------
        file_pattern: str
            Glob pattern matching HDF5 files (e.g., "/path/batch_*/segmented_rings_260831.h5")
        pmt_positions_file: str
            Location of an npz file containing the mapping from PMT IDs to CNN image pixel locations
        use_memmap: bool
            Whether to use memory mapping (not applicable for hierarchical structure, kept for API compatibility)
        one_indexed: bool
            Whether the PMT IDs in the H5 file are indexed starting at 1 (like SK tube numbers) or 0 (like WCSim PMT
            indexes). By default, zero-indexing is assumed.
        """
        self.file_pattern = file_pattern
        self.use_memmap = use_memmap
        self.one_indexed = one_indexed
        
        # Load PMT positions mapping
        self.pmt_positions = np.load(pmt_positions_file)["pmt_image_positions"].astype(int)
        self.data_size = np.array([np.max(self.pmt_positions, axis=0) + 1], dtype=int).flatten()
        
        # Find all matching HDF5 files
        self.h5_files = sorted(glob(file_pattern))
        if not self.h5_files:
            raise FileNotFoundError(f"No HDF5 files found matching pattern: {file_pattern}")
        
        print(f"Found {len(self.h5_files)} HDF5 files")
        
        # Build index of (file_idx, event_id, ring_id) tuples
        self.ring_index = []
        self._build_ring_index()
        
        print(f"Total rings (samples): {len(self.ring_index)}")
    
    def _build_ring_index(self):
        """
        Scan all HDF5 files and build an index of available rings.
        Each entry is (file_idx, event_id, ring_id).
        """
        for file_idx, h5_path in enumerate(self.h5_files):
            try:
                with h5py.File(h5_path, 'r') as h5_file:
                    # Iterate over all events in the file
                    for event_id in sorted(h5_file.keys()):
                        if not event_id.startswith('event_'):
                            continue
                        
                        event_group = h5_file[event_id]
                        
                        # Iterate over all rings in the event
                        for ring_id in sorted(event_group.keys()):
                            if not ring_id.startswith('ring_'):
                                continue
                            
                            self.ring_index.append((file_idx, event_id, ring_id))
            
            except Exception as e:
                print(f"Error reading {h5_path}: {e}")
                raise
    
    def _load_ring_data(self, file_idx, event_id, ring_id):
        """
        Load data for a specific ring from an HDF5 file.
        
        Parameters
        ----------
        file_idx: int
            Index of the HDF5 file
        event_id: str
            Event ID (e.g., 'event_000000')
        ring_id: str
            Ring ID (e.g., 'ring_0')
        
        Returns
        -------
        dict
            Dictionary containing ring data
        """
        h5_path = self.h5_files[file_idx]
        
        with h5py.File(h5_path, 'r') as h5_file:
            ring_group = h5_file[event_id][ring_id]
            
            # Load scalar attributes
            energy = float(ring_group['energy'][()])
            event_type = int(ring_group['event_type'][()])
            
            particle_dir_x = float(ring_group['particle_dir_x'][()])
            particle_dir_y = float(ring_group['particle_dir_y'][()])
            particle_dir_z = float(ring_group['particle_dir_z'][()])
            
            particle_start_x = float(ring_group['particle_start_x'][()])
            particle_start_y = float(ring_group['particle_start_y'][()])
            particle_start_z = float(ring_group['particle_start_z'][()])
            
            # Load array attributes
            tube_ids = np.array(ring_group['tube_ids'], dtype=np.int32)
            pmt_charge = np.array(ring_group['pmt_charge'], dtype=np.float32)
            pmt_time = np.array(ring_group['pmt_time'], dtype=np.float32)
            
            data_dict = {
                'energy': energy,
                'event_type': event_type,
                'particle_dir': np.array([particle_dir_x, particle_dir_y, particle_dir_z], dtype=np.float32),
                'particle_start': np.array([particle_start_x, particle_start_y, particle_start_z], dtype=np.float32),
                'tube_ids': tube_ids,
                'pmt_charge': pmt_charge,
                'pmt_time': pmt_time,
                'n_hits': len(tube_ids),
            }
            
            return data_dict
    
    def _convert_event_type_to_label(self, event_type):
        """
        Convert event_type to labels.
        
        Parameters
        ----------
        event_type: int
            Event type code (11, 13, etc.)
        
        Returns
        -------
        int
            Label (0, 1, 2, etc.)
        """
        if event_type == 11:
            return 1  # electron
        elif event_type == 13:
            return 2  # muon
        else:
            return 0  # gamma or unknown
    
    def _convert_direction_to_angles(self, direction_vector):
        """
        Convert 3D direction vector to zenith and azimuth angles.
        
        Parameters
        ----------
        direction_vector: ndarray of shape (3,)
            Direction vector [x, y, z]
        
        Returns
        -------
        ndarray of shape (2,)
            Angles [zenith, azimuth] in radians
        
        Notes
        -----
        Conversion follows watchmal.utils.math conventions:
        - zenith = arccos(z)  # Polar angle from z-axis
        - azimuth = atan2(y, x)  # Azimuth angle in xy-plane
        
        Returns angles in order [zenith, azimuth] to match direction_from_angles() 
        and angles_from_direction() in watchmal.utils.math
        """
        dx, dy, dz = direction_vector
        
        # Compute zenith angle (with clipping to avoid numerical errors)
        zenith = np.arccos(np.clip(dz, -1.0, 1.0))
        
        # Compute azimuth angle
        azimuth = np.arctan2(dy, dx)
        
        return np.array([zenith, azimuth], dtype=np.float32)
    
    def process_data(self, ring_data):
        """
        Convert ring data to normalized format matching CNNDataset structure.
        
        Parameters
        ----------
        ring_data: dict
            Dictionary containing raw ring data
        
        Returns
        -------
        dict
            Dictionary with converted and normalized data
        """
        processed = {}
        
        # 1. Energy (direct)
        processed['energy'] = np.array([ring_data['energy']], dtype=np.float32)
        
        # 2. Convert event_type to labels
        processed['label'] = np.array([self._convert_event_type_to_label(ring_data['event_type'])], dtype=np.int32)
        
        # 3. Position (particle_start -> positions)
        processed['position'] = ring_data['particle_start'].reshape(1, 1, 3).astype(np.float32)
        
        # 4. Convert direction to angles [zenith, azimuth]
        processed['angles'] = self._convert_direction_to_angles(ring_data['particle_dir']).reshape(1, 2).astype(np.float32)
        
        # 5-7. Hit data (direct mapping)
        processed['hit_pmt'] = ring_data['tube_ids']
        processed['hit_charge'] = ring_data['pmt_charge']
        processed['hit_time'] = ring_data['pmt_time']
        processed['n_hits'] = ring_data['n_hits']
        
        return processed
    
    def __len__(self):
        """Return total number of rings (samples) in the dataset."""
        return len(self.ring_index)
    
    def __getitem__(self, idx):
        """
        Get a ring sample by index.
        
        Parameters
        ----------
        idx: int
            Index of the sample
        
        Returns
        -------
        dict
            Dictionary containing ring data ready for conversion/training
        """
        file_idx, event_id, ring_id = self.ring_index[idx]
        
        ring_data = self._load_ring_data(file_idx, event_id, ring_id)
        processed_data = self.process_data(ring_data)
        
        # Add metadata
        processed_data['file_idx'] = file_idx
        processed_data['event_id'] = event_id
        processed_data['ring_id'] = ring_id
        processed_data['index'] = idx
        
        return processed_data
