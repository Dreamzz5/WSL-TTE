# -*- coding: utf-8 -*-
import json
import urllib
import math

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方




class Convert():
    def __init__(self):
        pass

    def convert(self, lng, lat):
        return lng, lat


class GCJ02ToWGS84(Convert):
    def __init__(self):
        super().__init__()

    def convert(self, lng, lat):
        """
        GCJ02(火星坐标系)转GPS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        if out_of_china(lng, lat):
            return [lng, lat]
        dlat = _transformlat(lng - 105.0, lat - 35.0)
        dlng = _transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return lng * 2 - mglng, lat * 2 - mglat


def out_of_range(lng, lat):
    """
    判断是否在研究范围内（西安）
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 108.9 and lng < 109 and lat > 34.2 and lat < 34.29)



