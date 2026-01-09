import numpy as np
from typing import Dict, List, Optional

def make_demo_world_points():
    lane_width = 3.5
    z0 = 0.0

    # =========================================================
    # 1) Main road (east-west, x direction)
    # =========================================================
    xs = np.linspace(-40, 40, 400)
    z_main = z0 * np.ones_like(xs)

    main_center = np.stack([xs, np.zeros_like(xs), z_main], axis=1)
    main_left   = np.stack([xs,  (lane_width/2)*np.ones_like(xs), z_main], axis=1)
    main_right  = np.stack([xs, -(lane_width/2)*np.ones_like(xs), z_main], axis=1)

    lane_center = np.vstack([main_center])
    lane_left   = np.vstack([main_left])
    lane_right  = np.vstack([main_right])
    
    # car_center = np.array([ -20.0, -1.0, 0.0 ])  # starting before intersection
    car_center = np.array([25.0, 1.0, 0.0])

    L, W, H = 4.5, 2.0, 1.5

    box = []
    for dz in [0.0, H]:
        for dx in [-L/2, L/2]:
            for dy in [-W/2, W/2]:
                box.append(car_center + np.array([dx, dy, dz]))

    car_box = np.array(box)

    return {
        "lane_center": lane_center,
        "lane_left": lane_left,
        "lane_right": lane_right,
        "car_box": car_box
    }

def shift_pts(items, dx, dy):
    if items is None:
        return None
    if isinstance(items, list):
        out=[]
        for a in items:
            if a is None: 
                continue
            b = np.asarray(a).copy()
            if b.ndim == 1:
                b = b.reshape(1,-1)
            b[:,0] += dx
            b[:,1] += dy
            out.append(b)
        return out
    b = np.asarray(items).copy()
    if b.ndim == 1:
        b = b.reshape(1,-1)
    b[:,0] += dx
    b[:,1] += dy
    return b

def make_pangyo_world_pts(
    area_half: float = 200.0,          # 전체 400m x 400m
    lane_width: float = 3.5,
    lanes_each_dir: int = 4,           # 한 방향 4차로 (총 8차로)
    sample_step: float = 0.5,
    z0: float = 0.0,
    ) -> Dict[str, object]:
    
    def _linspace_pts_1d(a: float, b: float, step: float) -> np.ndarray:
        n = max(2, int(np.ceil(abs(b - a) / step)) + 1)
        return np.linspace(a, b, n)

    def _make_straight_polyline(axis: str, s0: float, s1: float, offset: float) -> np.ndarray:
        ss = _linspace_pts_1d(s0, s1, sample_step)
        if axis == "x":
            x, y = ss, np.full_like(ss, offset)
        else:
            y, x = ss, np.full_like(ss, offset)
        z = np.full_like(ss, z0)
        return np.stack([x, y, z], axis=1)

    lane_center: List[np.ndarray] = []
    lane_left:   List[np.ndarray] = []
    lane_right:  List[np.ndarray] = []
    stop_line:   List[np.ndarray] = []

    # ---------------------------------------
    # 1) 도로 기본 파라미터
    # ---------------------------------------
    road_half_width = lanes_each_dir * lane_width 

    # ---------------------------------------
    # 2) E-W 도로 차선들 (진행방향: x)
    # ---------------------------------------    
    # y = +14.0 (북쪽 끝), y = -14.0 (남쪽 끝)
    lane_left.append(_make_straight_polyline("x", -area_half, area_half, road_half_width))
    lane_right.append(_make_straight_polyline("x", -area_half, area_half, -road_half_width))

    # y 좌표: [-14, -10.5, -7.0, -3.5, 0, 3.5, 7.0, 10.5, 14]
    lane_indices = np.arange(-lanes_each_dir, lanes_each_dir + 1)
    for i in lane_indices:
        y_offset = i * lane_width
        line = _make_straight_polyline("x", -area_half, area_half, y_offset)
        lane_center.append(line)

    # ---------------------------------------
    # 3) 정지선 (stop line)
    # ---------------------------------------
    # y=0에서 y=-7까지의 세로선 (x 좌표는 적절한 위치에 설정)
    stop_x = 0.0  # x 좌표 (필요시 조정 가능)
    stop_line.append(np.array([
        [stop_x, 0.0, z0],
        [stop_x, -7.0, z0]
    ]))

    # ---------------------------------------
    # 4) 선행차 (우측 첫 번째 차선 배치 및 크기 2배)
    # ---------------------------------------
    L, W, H = 4.5, 3.5, 3.0
    
    # 위치: 우측 첫 번째 차선의 중심 (중앙선 기준 -1.75m)
    # lane_width가 3.5일 때 -1.75 지점이 1차로(우측) 중심입니다.
    car_y = -lane_width * 0.5
    car_center = np.array([30.0, car_y, z0]) 
    
    box = []
    for dz in [0.0, H]:
        for dx in [-L/2, L/2]:
            for dy in [-W/2, W/2]:
                # car_center(차량 중심)를 기준으로 8개의 꼭짓점 생성
                box.append(car_center + np.array([dx, dy, dz]))
                
    car_box = np.array(box)

    return {
        "lane_center": lane_center,
        "lane_left": lane_left,
        "lane_right": lane_right,
        "car_box": car_box,
        "stop_line": stop_line,
    }