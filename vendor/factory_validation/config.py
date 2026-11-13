from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


NEWEXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
FACTORY_DATA_DIR = NEWEXPERIMENT_ROOT / "factory data"
OUTPUT_ROOT = NEWEXPERIMENT_ROOT / "factory_validation"
CONFIG_DIR = OUTPUT_ROOT / "config"
DATA_DIR = OUTPUT_ROOT / "data"
RESULTS_DIR = OUTPUT_ROOT / "results"
PICTURE_DIR = OUTPUT_ROOT / "picture"


@dataclass(frozen=True)
class PathDefinition:
    index: int
    pair_1based: Tuple[int, int]
    length_chord_m: float
    length_effective_m: float
    source: str

    @property
    def name(self) -> str:
        return f"Path{self.index}"

    @property
    def pair_0based(self) -> Tuple[int, int]:
        return (self.pair_1based[0] - 1, self.pair_1based[1] - 1)

    def to_dict(self) -> dict:
        return asdict(self)


_PAIRS = [
    (1, 6), (2, 5), (3, 1), (4, 8), (5, 3), (6, 2),
    (7, 1), (8, 1), (1, 5), (2, 4), (3, 7), (4, 6),
    (5, 8), (6, 7), (7, 2), (8, 2), (1, 4), (2, 3),
    (3, 6), (4, 5), (5, 7), (6, 8), (7, 4), (8, 3),
]

_SENSOR_COL1 = [
    14.9010, 14.6689, 12.1192, 15.4003, 10.2858, 14.9530,
    10.3189, 6.0008, 16.4735, 10.9075, 15.3657, 9.4359,
    14.8051, 8.1400, 13.7391, 10.8762, 14.7656, 6.9156,
    12.6296, 5.9437, 12.1792, 11.7097, 14.6001, 14.6001,
]

_SENSOR_COL2 = [
    15.7010, 15.4689, 12.9192, 16.2003, 11.0858, 15.7530,
    11.1189, 5.8008, 17.2735, 11.7075, 16.1657, 10.2359,
    15.6051, 8.9400, 14.5391, 11.6762, 15.5656, 7.7156,
    13.4296, 6.7437, 12.9792, 12.5097, 15.4001, 15.4001,
]

_HARDWARE_LENGTHS = [
    14.0021, 14.0010, 10.0133, 15.2521, 11.2749, 14.7113,
    11.1080, 5.6203, 14.7746, 11.4681, 15.0929, 9.7059,
    13.7451, 5.8892, 13.6723, 9.7536, 14.0514, 5.8600,
    13.7210, 5.7499, 9.7671, 11.3232, 13.6030, 13.6002,
]


def _make_paths(chord_lengths: Iterable[float], effective_lengths: Iterable[float], source: str) -> List[PathDefinition]:
    return [
        PathDefinition(
            index=i + 1,
            pair_1based=_PAIRS[i],
            length_chord_m=float(chord),
            length_effective_m=float(effective),
            source=source,
        )
        for i, (chord, effective) in enumerate(zip(chord_lengths, effective_lengths))
    ]


SENSOR_CONFIG_PATHS = _make_paths(
    _SENSOR_COL1,
    _SENSOR_COL2,
    "sensor_config_docx_col1_chord_col2_effective",
)

SENSOR_CONFIG_CHORD_ONLY_PATHS = _make_paths(
    _SENSOR_COL1,
    _SENSOR_COL1,
    "sensor_config_docx_col1_chord_and_effective",
)

SENSOR_CONFIG_EFFECTIVE_ONLY_PATHS = _make_paths(
    _SENSOR_COL2,
    _SENSOR_COL2,
    "sensor_config_docx_col2_chord_and_effective",
)

HARDWARE_PATHS = _make_paths(
    _HARDWARE_LENGTHS,
    _HARDWARE_LENGTHS,
    "acoustic_hardware_docx",
)

PATH_SETS = {
    "sensor_config": SENSOR_CONFIG_PATHS,
    "sensor_config_chord_only": SENSOR_CONFIG_CHORD_ONLY_PATHS,
    "sensor_config_effective_only": SENSOR_CONFIG_EFFECTIVE_ONLY_PATHS,
    "hardware": HARDWARE_PATHS,
}

PATH_COLUMNS = [f"Path{i}" for i in range(1, 25)]
AREA_COLUMNS = [f"Area{i}" for i in range(1, 25)]


def path_pairs_0based(paths: Iterable[PathDefinition]) -> List[Tuple[int, int]]:
    return [path.pair_0based for path in paths]


def chord_lengths(paths: Iterable[PathDefinition]):
    return [path.length_chord_m for path in paths]


def effective_lengths(paths: Iterable[PathDefinition]):
    return [path.length_effective_m for path in paths]
