#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
china_xml_tiff_to_isce2.py

功能：
1. 国产 SAR XML -> ISCE2 sensor.xml + orbit.xml
2. 自动补全 pulseDuration / chirpSlope
3. 国产 TIFF SLC -> ISCE2 可读 SLC（二进制）
"""

import sys
import xml.etree.ElementTree as ET
import numpy as np
from math import fabs
from osgeo import gdal

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

def find_text(root, tags):
    for t in tags:
        e = root.find(f".//{t}")
        if e is not None and e.text:
            return e.text.strip()
    return None


# ------------------------------------------------------------
# pulseDuration / chirpSlope 估算
# ------------------------------------------------------------

def estimate_pulse_duration(meta):

    if meta.get("chirpDuration"):
        return float(meta["chirpDuration"])

    if meta.get("rangeBandwidth"):
        print("[WARNING] pulseDuration fallback: 1/B")
        return 1.0 / float(meta["rangeBandwidth"])

    raise RuntimeError("Cannot estimate pulseDuration")


def estimate_chirp_slope(meta, pulse_dur):

    if meta.get("chirpSlope"):
        K = float(meta["chirpSlope"])
        if abs(K) > 1e6:
            return K

    if meta.get("chirp_rate") and meta.get("rangeBandwidth"):
        sign = float(meta["chirp_rate"])
        B = float(meta["rangeBandwidth"])
        return sign * B / pulse_dur

    raise RuntimeError("Cannot estimate chirpSlope")


# ------------------------------------------------------------
# 轨道解析
# ------------------------------------------------------------

def parse_orbit(root):

    svs = []

    for sv in root.findall(".//stateVector") + root.findall(".//OrbitStateVector"):

        def get(*n):
            for x in n:
                e = sv.find(x)
                if e is not None and e.text:
                    return e.text.strip()
            return None

        rec = dict(
            time=get("time", "UTC"),
            x=get("x", "PositionX"),
            y=get("y", "PositionY"),
            z=get("z", "PositionZ"),
            vx=get("vx", "VelocityX"),
            vy=get("vy", "VelocityY"),
            vz=get("vz", "VelocityZ"),
        )

        if None not in rec.values():
            svs.append(rec)

    if len(svs) < 5:
        raise RuntimeError("Orbit vectors < 5")

    return svs


# ------------------------------------------------------------
# TIFF -> SLC
# ------------------------------------------------------------

def tiff_to_slc(tiff, slc_out):

    ds = gdal.Open(tiff, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("Cannot open TIFF")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    if np.iscomplexobj(arr):
        slc = arr.astype(np.complex64)
    else:
        slc = arr.astype(np.float32)

    slc.tofile(slc_out)
    print(f"[OK] SLC written: {slc_out}")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def convert(xml_in, tiff_in, sensor_out, orbit_out, slc_out):

    root = ET.parse(xml_in).getroot()

    meta = {
        "prf": find_text(root, ["PRF"]),
        "rangeSamplingRate": find_text(root, ["rangeSamplingRate"]),
        "rangeBandwidth": find_text(root, ["rangeBandwidth", "chirpBandwidth"]),
        "chirpSlope": find_text(root, ["chirpSlope", "FMRate"]),
        "chirp_rate": find_text(root, ["chirp_rate"]),
        "chirpDuration": find_text(root, ["chirpDuration"]),
        "samplesPerPulse": find_text(root, ["samplesPerPulse"]),
        "radarFrequency": find_text(root, ["radarFrequency"]),
        "sensingStart": find_text(root, ["sensingStart"]),
        "sensingStop": find_text(root, ["sensingStop"]),
    }

    pulse_dur = estimate_pulse_duration(meta)
    chirp_slope = estimate_chirp_slope(meta, pulse_dur)

    # -------- sensor.xml --------
    sensor = ET.Element("sensor")

    def prop(n, v):
        e = ET.SubElement(sensor, "property", name=n)
        e.text = str(v)

    prop("prf", meta["prf"])
    prop("rangeSamplingRate", meta["rangeSamplingRate"])
    prop("pulseDuration", f"{pulse_dur:.12e}")
    prop("chirpSlope", f"{chirp_slope:.6e}")
    prop("radarFrequency", meta["radarFrequency"])
    prop("rangeBandwidth", meta["rangeBandwidth"])

    ET.ElementTree(sensor).write(sensor_out, encoding="utf-8", xml_declaration=True)

    # -------- orbit.xml --------
    orbit = ET.Element("orbit")
    for sv in parse_orbit(root):
        e = ET.SubElement(orbit, "stateVector")
        ET.SubElement(e, "time").text = sv["time"]
        ET.SubElement(e, "position").text = f"{sv['x']} {sv['y']} {sv['z']}"
        ET.SubElement(e, "velocity").text = f"{sv['vx']} {sv['vy']} {sv['vz']}"

    ET.ElementTree(orbit).write(orbit_out, encoding="utf-8", xml_declaration=True)

    # -------- TIFF -> SLC --------
    tiff_to_slc(tiff_in, slc_out)

    print("[DONE] ISCE2 inputs ready")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage:")
        print("  china_xml_tiff_to_isce2.py meta.xml image.tiff sensor.xml orbit.xml image.slc")
        sys.exit(1)

    convert(*sys.argv[1:])

