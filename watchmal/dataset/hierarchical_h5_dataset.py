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
    Compatible with CNNDataset interface for training and evaluation.
    
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
    
    def __init__(self, file_pattern, pmt_positions_file, use_memmap=False, one_indexed=False, 
                 mask_pmts=None, channel_scale_factor=None, channel_scale_offset=None,
                 use_times=True, use_charges=True, use_isHit=False, use_positions=False,
                 use_orientations=False, geometry_file=None, use_invalid_value=False,
                 use_median_unhit_times=False, use_log_charge=False, use_padding=False,
                 padding_to_fixed_dimension=None):
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
        mask_pmts: list of int
            List of PMT IDs to mask out from all data (None by default)
        channel_scale_factor: dict of float
            Dictionary with keys corresponding to channels and values contain the factors to divide that channel.
        channel_scale_offset: dict of float
            Dictionary with keys corresponding to channels and values contain the offsets to subtract from that channel.
        use_times: bool
            Whether to use PMT hit times as one of the initial CNN image channels. True by default.
        use_charges: bool
            Whether to use PMT hit charges as one of the initial CNN image channels. True by default.
        use_isHit: bool
            Whether to use a channel to tag the PMT hit or not.
        use_positions: bool
            Whether to use three channels to add the real positions info of PMTs.
        use_orientations: bool
            Whether to use three channels to add the real orientations info of PMTs.
        geometry_file: string
            Location of an npz file containing the real positions and orientations info.
        use_invalid_value: bool
            Whether to set all the channel of unhit as an invalid value (like -100).
        use_median_unhit_times: bool
            Whether to set unhit times to the median value of the normalised hit times
        use_log_charge: bool
            Whether to logarithmically transform the charge.
        use_padding: bool
            Whether to pad the data to a fixed dimension (default: False).
        padding_to_fixed_dimension: list of int
            If use_padding is True, this specifies the fixed dimension to which the data will be padded.
        """
        self.file_pattern = file_pattern
        self.use_memmap = use_memmap
        self.one_indexed = one_indexed
        self.mask_pmts = mask_pmts
        
        # Channel configuration (for CNN processing compatibility)
        self.use_times = use_times
        self.use_charges = use_charges
        self.use_isHit = use_isHit
        self.use_positions = use_positions
        self.use_orientations = use_orientations
        self.use_invalid_value = use_invalid_value
        self.use_median_unhit_times = use_median_unhit_times
        self.use_log_charge = use_log_charge
        
        if channel_scale_offset is None:
            channel_scale_offset = {}
        self.scale_offset = channel_scale_offset
        if channel_scale_factor is None:
            channel_scale_factor = {}
        self.scale_factor = channel_scale_factor
        
        # Load PMT positions mapping
        self.pmt_positions = np.load(pmt_positions_file)["pmt_image_positions"].astype(int)
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        if use_padding and padding_to_fixed_dimension is not None:
            self.data_size = np.array(padding_to_fixed_dimension)
        
        # Geometry data (optional)
        if use_positions and geometry_file:
            self.real_3Dpositions = np.load(geometry_file)["position"]
        else:
            self.real_3Dpositions = None
        if use_orientations and geometry_file:
            self.real_3Dorientations = np.load(geometry_file)["orientation"]
        else:
            self.real_3Dorientations = None
        
        # Find all matching HDF5 files
        self.h5_files = sorted(glob(file_pattern))
        if not self.h5_files:
            raise FileNotFoundError(f"No HDF5 files found matching pattern: {file_pattern}")
        
        print(f"Found {len(self.h5_files)} HDF5 files")
        
        # Build index of (file_idx, event_id, ring_id) tuples
        self.ring_index = []
        self._build_ring_index()
        
        print(f"Total rings (samples): {len(self.ring_index)}")
        
        # Instance variables for hit data (set during __getitem__)
        self.event_hit_pmts = None
        self.event_hit_charges = None
        self.event_hit_times = None
    
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
        Convert ring data to normalized format matching H5CommonDataset structure.
        
        Parameters
        ----------
        ring_data: dict
            Dictionary containing raw ring data
        
        Returns
        -------
        dict
            Dictionary with converted and normalized data in format matching CNNDataset
        """
        processed = {}
        
        # 1. Energy (direct) - shape (1,) to match H5CommonDataset format
        processed['energies'] = np.array([ring_data['energy']], dtype=np.float32)
        
        # 2. Convert event_type to labels - shape (1,)
        processed['labels'] = np.array([self._convert_event_type_to_label(ring_data['event_type'])], dtype=np.int32)
        
        # 3. Position (particle_start -> positions) - shape (1, 1, 3)
        processed['positions'] = ring_data['particle_start'].reshape(1, 1, 3).astype(np.float32)
        
        # 4. Convert direction to angles [zenith, azimuth] - shape (1, 2)
        processed['angles'] = self._convert_direction_to_angles(ring_data['particle_dir']).reshape(1, 2).astype(np.float32)
        
        # 5-7. Hit data (direct mapping) - keep as arrays for processing
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
            Dictionary containing ring data ready for CNN processing (compatible with CNNDataset format)
        """
        file_idx, event_id, ring_id = self.ring_index[idx]
        
        ring_data = self._load_ring_data(file_idx, event_id, ring_id)
        processed_data = self.process_data(ring_data)
        
        # Set instance variables for hit data (used by CNN processing)
        self.event_hit_pmts = processed_data['hit_pmt']
        self.event_hit_charges = processed_data['hit_charge']
        self.event_hit_times = processed_data['hit_time']
        
        # Apply PMT masking if specified
        if self.mask_pmts is not None:
            mask = np.isin(self.event_hit_pmts, self.mask_pmts, invert=True)
            self.event_hit_pmts = self.event_hit_pmts[mask]
            self.event_hit_charges = self.event_hit_charges[mask]
            self.event_hit_times = self.event_hit_times[mask]
        
        # Build output dict with targets in format matching H5CommonDataset
        data_dict = {
            'energies': processed_data['energies'],
            'labels': processed_data['labels'],
            'positions': processed_data['positions'],
            'angles': processed_data['angles'],
        }
        
        # Add metadata
        data_dict['file_idx'] = file_idx
        data_dict['event_id'] = event_id
        data_dict['ring_id'] = ring_id
        data_dict['indices'] = idx
        
        return data_dict
