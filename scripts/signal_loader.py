import functools
from contextlib import redirect_stdout
from os import path

import kikuchipy as kp
from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.signals.ebsd import EBSD, LazyEBSD
import numpy as np
import orix
from orix.crystal_map import CrystalMap
from diffsims.crystallography import ReciprocalLatticeVector

from utils import SettingFile
from utils.setting_file import get_setting_file_bottom_top
from utils.threads.thdout import ThreadedOutput


class crystalMap:

    def __init__(self, dataset: CrystalMap, crystal_map_path: str, compute_all: bool = False):
        """
        A custom crystalMap class storing the CrystalMap, dataset type, a phase id array, navigation shape, 
        corresponding EBSD dataset, navigators and Miller indices for diffraction planes used in geometrical simulation.

        Parameters
        ----------
        dataset : orix.crystal_map.CrystalMap
            Crystal map containing the phases and unit cell rotations, generated from either Hough or Dictionary indexing.
        crystal_map_path : str
            The crystal map file path

        Notes
        -----
        Depending on the crystal map different navigators are calculated.

        Hough indexing: \n
        - Single phase: Inverse pole figure
        - Multi-phase: Inverse pole figure, Phase map

        Dictionary indexing: \n
        - Single phase: Inverse pole figure, Normalized cross-correlation map, Orientation similarity metric map
        - Single phase with refined orientations: Inverse pole figure, Normalized cross-correlation map'

        - Multi-phase: Inverse pole figure, Phase map, Normalized cross-correlation map, Orientation similarity metric map
        - Multi-phase with refined orientations: Inverse pole figure, Phase map, Normalized cross-correlation map
        """

        self.crystal_map = dataset
        self.datatype = "crystal map"
        if compute_all:
            self.phase_id_array = self.crystal_map.get_map_data("phase_id")
            self.nav_shape = self.crystal_map.shape[::-1]
            self.ebsd = self.ebsd_signal(crystal_map_path)
            thdout = ThreadedOutput()
            with redirect_stdout(thdout):
                if len(self.crystal_map.phases.ids) > 1:
                    self.navigator = {
                        "Inverse pole figure": 0,
                        "Phase map": 1,
                    }
                else:
                    self.navigator = {
                        "Inverse pole figure": 0,
                    }
                if "scores" in self.crystal_map.prop:
                    self.navigator["Normalized cross-correlation"] = 2
                if self.crystal_map.rotations_per_point > 1:
                    self.navigator["Orientation similarity metric"] = 3

                self.hkl = self.hkl_simulation()

            self.ebsd_detector = self.detector(crystal_map_path)

    def ebsd_signal(self, crystal_map_path: str) -> LazyEBSD:
        """
        Loads the corresponding EBSD dataset which the crystal map was generated from. The crystal_map_path is used to get the name of the EBSD dataset from indexing_parameters.txt stored together with the crystal map.

        Parameters
        ----------
        crystal_map_path : str
            The crystal map file path

        Returns
        -------
        ebsd_signal : LazyEBSD
            Scan of Electron Backscatter Diffraction (EBSD) patterns

        """
        try:
            parameter_file, parameter_file_path = get_setting_file_bottom_top(
                crystal_map_path, "indexing_parameters.txt", return_dir_path=True
            )
        except FileNotFoundError as e:
            raise e

        ebsd_signal_name = parameter_file.read("Pattern name")
        ebsd_signal = kp.load(
            path.join(path.dirname(parameter_file_path), ebsd_signal_name)
        )

        return ebsd_signal

    def hkl_simulation(self) -> list:
        """
        Returns a list of Miller indices corresponding to the four strongest diffracting lattice planes for each phase in the CrystalMap.

        Returns
        -------
        hkl_simulations : list
            A nested list of Miller indices

        Notes
        -----
        Calculations of reciprocal lattice vectors and structure factors are done using `diffsim`.
        """
        hkl_simulations = []

        for i, ph in self.crystal_map.phases:
            phase_lattice = ph.structure.lattice
            phase_lattice.setLatPar(
                phase_lattice.a * 10, phase_lattice.b * 10, phase_lattice.c * 10
            )  # diffsim uses ångstrøm and not nm for lattice parameters

            if i != -1:
                rlv = ReciprocalLatticeVector.from_min_dspacing(ph, 0.7)

                rlv.sanitise_phase()
                rlv = rlv.unique(use_symmetry=True)
                rlv.calculate_structure_factor("lobato")

                structure_factor = abs(rlv.structure_factor)
                order = np.argsort(structure_factor)[::-1]
                rlv = rlv[order[0:4]]
                rlv = rlv.symmetrise()
                hkl_simulations.append(rlv)

        return hkl_simulations

    def detector(self, crystal_map_path) -> EBSDDetector:
        """
        Loads the detector parameters stored together with the CrystalMap. The detector parameters are stored in the file detector.txt which is stored together with the crystal map.

        Parameters
        ----------
        crystal_map_path : str
            Crystal map file path.

        Returns
        -------
        detector : kikuchipy.detectors.ebsd_detector.EBSDDetector
            An EBSD detector class storing its shape, pixel size, binning factor, detector tilt, sample tilt and projection center (PC).
        """
        detector = kp.detectors.EBSDDetector.load(
            path.join(
                get_setting_file_bottom_top(
                    crystal_map_path, "detector.txt", return_dir_path=True
                )[-1],
                "detector.txt",
            )
        )
        ebsd_signal = self.ebsd_signal(crystal_map_path)
        detector.shape = ebsd_signal.axes_manager.signal_shape

        return detector

    def compute_navigator(self, nav_num: int = 0) -> np.ndarray:
        """
        Computes a navigator to be used in the navigator view in the signal navigation widget.\n

        Parameters
        ----------
        nav_num : int
            The navigator id

        Returns
        -------
        navigator: numpy.ndarray
            Returns the selected navigator.

        Notes
        -----
        Navigator ids: \n
        0 = inverse pole figure, 1 = phase map, 2 = ncc_map, 3 = osm_map
        """

        if nav_num == 0:
            navigator = self.inverse_pole_figure
            return navigator

        if nav_num == 1:
            navigator = self.phase_map
            return navigator

        if nav_num == 2:
            navigator = self.normalized_cross_correlation_map
            return navigator

        if nav_num == 3:
            navigator = self.orientation_simliarity_metric
            return navigator

    @functools.cached_property
    def inverse_pole_figure(self) -> np.ndarray:
        """
        Inverse pole figure map. Used as default navigator for crystal maps.
        """
        if self.crystal_map.phases.ids[0] == -1:
            phase_id = self.crystal_map.phases.ids[1]
        else:
            phase_id = self.crystal_map.phases.ids[0]

        ckey = orix.plot.IPFColorKeyTSL(
            self.crystal_map.phases[phase_id].point_group)
        rgb_all = np.zeros((self.crystal_map.size, 3))
        for i, phase in self.crystal_map.phases:
            if i != -1:
                rgb_i = ckey.orientation2color(
                    self.crystal_map[phase.name].orientations
                )
                rgb_all[self.crystal_map.phase_id == i] = rgb_i

        rgb_all = rgb_all.reshape(self.crystal_map.shape + (3,))

        return rgb_all

    @functools.cached_property
    def phase_map(self) -> np.ndarray:
        """
        Phase map for crystal maps with more than one phase.
        """
        thdout = ThreadedOutput()
        with redirect_stdout(thdout):
            phase_id = self.crystal_map.get_map_data("phase_id")
            unique_phase_ids = np.unique(phase_id[~np.isnan(phase_id)])
            phase_map = np.ones(phase_id.shape + (3,))
            for i, color in zip(
                unique_phase_ids, self.crystal_map.phases_in_data.colors_rgb
            ):
                mask = phase_id == int(i)
                phase_map[mask] = phase_map[mask] * color

            phase_map = np.squeeze(phase_map)

        return phase_map

    @functools.cached_property
    def normalized_cross_correlation_map(self) -> np.ndarray:
        """
        Normalized cross-correlation for crystal maps from dictionary indexing.
        """
        if self.crystal_map.rotations_per_point > 1:
            ncc_map = self.crystal_map.scores[:, 0].reshape(
                *self.crystal_map.shape)
        else:
            ncc_map = self.crystal_map.get_map_data("scores")

        return ncc_map

    @functools.cached_property
    def orientation_simliarity_metric(self) -> np.ndarray:
        """
        Orientation similarity metric for unrefined crystal maps from dictionary indexing.
        """
        osm_map = kp.indexing.orientation_similarity_map(self.crystal_map)

        return osm_map


class EBSDDataset:

    def __init__(self, dataset):
        """
        A custom EBSDDataset class storing the EBSD, dataset type, navigation shape and navigators.

        Parameters
        ----------
        dataset : EBSD | LazyEBSD
            ebsd dataset

        Notes
        -----
        Available navigators for EBSD datasets are the Mean intensity map (default), Image quality map and Virtual BSE map.
        """

        self.ebsd = dataset
        self.datatype = "ebsd_dataset"
        self.nav_shape = self.ebsd.axes_manager.navigation_shape
        self.navigator = {
            "Mean intensity map": 0,
            "Image quality map": 1,
            "VBSE map": 2,
        }

    def compute_navigator(self, nav_num: int = 0):
        """
        Computes a navigator to be used in the navigator view in the signal navigation widget.\n
        
        Parameters
        ----------
        nav_num : int
            The navigator id

        Returns
        -------
        navigator: numpy.ndarray
            Returns the selected navigator.

        Notes
        -----
        Navigator ids: \n
        0 = inverse pole figure, 1 = image quality, 2 = virtual bse image
        """

        if nav_num == 0:
            navigator = self.mean_intensity
            return navigator
        if nav_num == 1:
            navigator = self.image_quality
            return navigator
        if nav_num == 2:
            navigator = self.virtual_bse
            return navigator

    @functools.cached_property
    def mean_intensity(self):
        """
        Mean intensity map
        """
        mean_intensity_map = self.ebsd.mean(axis=(2, 3))
        return mean_intensity_map

    @functools.cached_property
    def image_quality(self):
        """
        Image quality (IQ) map
        """
        return self.ebsd.get_image_quality().compute()

    @functools.cached_property
    def virtual_bse(self):
        """
        Virtual BSE (VBSE) image
        """
        vbse_gen = kp.generators.VirtualBSEGenerator(self.ebsd)
        vbse_map = vbse_gen.get_rgb_image(r=(3, 1), b=(3, 2), g=(3, 3))
        vbse_map.change_dtype("uint8")
        vbse_map = vbse_map.data
        return vbse_map
