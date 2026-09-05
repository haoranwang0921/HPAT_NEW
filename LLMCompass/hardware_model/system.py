from hardware_model.device import Device, device_dict
from hardware_model.interconnect import InterConnectModule, interconnect_module_dict
from typing import List


class System:
    def __init__(self, pcb_module: Device, interconnect: InterConnectModule) -> None:
        self.device = pcb_module
        self.interconnect = interconnect


system_dict = {
    "A100_4_fp16": System(
        device_dict["A100_80GB_fp16"],
        interconnect_module_dict["NVLinkV3_FC_4"],
    ),
    "TPUv3_8": System(device_dict["TPUv3"], interconnect_module_dict["TPUv3Link_8"]),
    # HPAT_Hybrid: photonic accelerator placeholder (Plan Section 11 G2).
    # Requires photonic_module to be attached at runtime via SimPhonyBackend.
    "HPAT_Hybrid": None,
}
