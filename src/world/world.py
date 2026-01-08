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
    area_half: float = 200.0,          # -> 전체 400m x 400m
    intersection_half: float = 25.0,    # 교차로 중심부 반경(대략 50m x 50m 박스)
    lane_width: float = 3.5,
    lanes_each_dir: int = 2,           # 한 방향 차로 수 (예: 2면 왕복 4차로)
    sample_step: float = 0.5,          # 직선 샘플링 간격(미터)
    z0: float = 0.0,
    add_turn_guides: bool = False,      # 좌/우회전 가이드(곡선) centerline 추가
) -> Dict[str, object]:
    """
    4-way 교차로(판교 느낌) 월드 포인트를 생성.
    좌표계:
      - 원점(0,0)이 교차로 중심
      - +x: 동쪽, +y: 북쪽
      - z는 평면 z0

    Returns (기존과 호환):
      {
        "lane_center": List[np.ndarray],  # 각 polyline: (N,3)
        "lane_left":   List[np.ndarray],
        "lane_right":  List[np.ndarray],
        "car_box":     np.ndarray | None, # (8,3)
        "stop_line":   List[np.ndarray],  # (K,3) 짧은 선분 polyline
      }
    """

    def _linspace_pts_1d(a: float, b: float, step: float) -> np.ndarray:
        n = max(2, int(np.ceil(abs(b - a) / step)) + 1)
        return np.linspace(a, b, n)

    def _make_straight_polyline(axis: str, s0: float, s1: float, offset: float) -> np.ndarray:
        """
        axis='x': x가 진행방향(동-서), y는 offset 고정
        axis='y': y가 진행방향(남-북), x는 offset 고정
        """
        ss = _linspace_pts_1d(s0, s1, sample_step)
        if axis == "x":
            x = ss
            y = np.full_like(ss, offset)
        elif axis == "y":
            y = ss
            x = np.full_like(ss, offset)
        else:
            raise ValueError("axis must be 'x' or 'y'")
        z = np.full_like(ss, z0)
        return np.stack([x, y, z], axis=1)

    def _offset_boundary(poly: np.ndarray, normal_xy: np.ndarray, dist: float) -> np.ndarray:
        # 직선 polyline 가정(이번 생성에서는 직선/원호 위주라 충분)
        # normal_xy는 (2,) 방향 벡터(정규화되어있지 않아도 됨)
        n = normal_xy / (np.linalg.norm(normal_xy) + 1e-12)
        out = poly.copy()
        out[:, 0] += n[0] * dist
        out[:, 1] += n[1] * dist
        return out

    def _make_box_polygon(cx: float, cy: float, hx: float, hy: float) -> np.ndarray:
        # 닫힌 사각형(마지막 점 = 첫 점)
        pts = np.array([
            [cx - hx, cy - hy, z0],
            [cx + hx, cy - hy, z0],
            [cx + hx, cy + hy, z0],
            [cx - hx, cy + hy, z0],
            [cx - hx, cy - hy, z0],
        ], dtype=float)
        return pts

    def _make_arc(center: np.ndarray, r: float, th0: float, th1: float, step_m: float) -> np.ndarray:
        # 원호 길이 기준으로 샘플 개수 결정
        arc_len = abs(th1 - th0) * r
        n = max(10, int(np.ceil(arc_len / step_m)) + 1)
        th = np.linspace(th0, th1, n)
        x = center[0] + r * np.cos(th)
        y = center[1] + r * np.sin(th)
        z = np.full_like(x, z0)
        return np.stack([x, y, z], axis=1)

    lane_center: List[np.ndarray] = []
    lane_left:   List[np.ndarray] = []
    lane_right:  List[np.ndarray] = []
    stop_line:   List[np.ndarray] = []

    # ---------------------------------------
    # 1) 도로 기본 파라미터
    # ---------------------------------------
    # 왕복 도로: 중심선 기준으로 한쪽(예: +y 방향 진행 lanes)과 반대쪽 lanes를 만들 것
    # E-W 도로(진행방향 x): lane의 법선은 y축
    # N-S 도로(진행방향 y): lane의 법선은 x축
    road_half_width = lanes_each_dir * lane_width  # 중심선~도로 가장자리(차로만 고려)

    # 직선 구간: 교차로 중심부는 비우고 approach만 생성
    # 예) 서쪽 접근: x in [-area_half, -intersection_half], 동쪽 접근: [intersection_half, area_half]
    x_w0, x_w1 = -area_half, area_half
    x_e0, x_e1 = -area_half, area_half
    y_s0, y_s1 = -area_half, area_half
    y_n0, y_n1 = -area_half, area_half

    # ---------------------------------------
    # 2) E-W 도로 차선들 (진행방향: x)
    #   - 위쪽(+y) : 동쪽(+x) 진행(가정)
    #   - 아래쪽(-y): 서쪽(-x) 진행(가정)
    # ---------------------------------------
    # 차선 center offset: (0.5, 1.5, ...) * lane_width 형태로 배치
    offsets_one_side = (np.arange(lanes_each_dir) + 0.5) * lane_width  # 1차선,2차선...

    # +y측 차선들 (y = +offset)
    for off in offsets_one_side:
        c = _make_straight_polyline("x", -area_half, area_half, +off)
        lane_center.append(c)
        lane_left.append(_offset_boundary(c, np.array([0.0, 1.0]), +lane_width/2))
        lane_right.append(_offset_boundary(c, np.array([0.0, 1.0]), -lane_width/2))

    # -y측 차선들 (y = -offset)
    for off in offsets_one_side:
        c = _make_straight_polyline("x", -area_half, area_half, -off)
        lane_center.append(c)
        lane_left.append(_offset_boundary(c, np.array([0.0, 1.0]), +lane_width/2))
        lane_right.append(_offset_boundary(c, np.array([0.0, 1.0]), -lane_width/2))

    # ---------------------------------------
    # 3) N-S 도로 차선들 (진행방향: y)
    #   - 오른쪽(+x): 북쪽(+y) 진행(가정)
    #   - 왼쪽(-x)  : 남쪽(-y) 진행(가정)
    # ---------------------------------------
    for off in offsets_one_side:
        c = _make_straight_polyline("y", -area_half, area_half, +off)

    for off in offsets_one_side:
        c = _make_straight_polyline("y", -area_half, area_half, -off)

    # ---------------------------------------
    # 4) 교차로 내부 회전(가이드) 차선(선택)
    #    - 간단히 quarter-circle로 좌/우회전 가이드 생성
    # ---------------------------------------
    if add_turn_guides:
        # 회전 반경: 안쪽차선 기준으로 intersection_half - 여유
        r = max(8.0, intersection_half - lane_width * 1.0)

        # 4개 코너에서 우회전(예시) 가이드
        # (서->북), (북->동), (동->남), (남->서) 등
        # center는 각 코너 근처로 잡고 90도 원호를 그림
        # 서->북: 중심 (-intersection_half, +intersection_half)
        c = _make_arc(center=np.array([-intersection_half, +intersection_half]),
                      r=r, th0=-0.0*np.pi, th1=+0.5*np.pi, step_m=sample_step)
        lane_center.append(c)
        lane_left.append(_offset_boundary(c, np.array([-np.sin(0.25*np.pi), np.cos(0.25*np.pi)]), +lane_width/2))
        lane_right.append(_offset_boundary(c, np.array([-np.sin(0.25*np.pi), np.cos(0.25*np.pi)]), -lane_width/2))

        # 북->동: 중심 (+intersection_half, +intersection_half)
        c = _make_arc(center=np.array([+intersection_half, +intersection_half]),
                      r=r, th0=+0.5*np.pi, th1=+0.0*np.pi, step_m=sample_step)
        lane_center.append(c)
        lane_left.append(_offset_boundary(c, np.array([1.0, 0.0]), +lane_width/2))
        lane_right.append(_offset_boundary(c, np.array([1.0, 0.0]), -lane_width/2))

        # 동->남: 중심 (+intersection_half, -intersection_half)
        c = _make_arc(center=np.array([+intersection_half, -intersection_half]),
                      r=r, th0=+1.0*np.pi, th1=+0.5*np.pi, step_m=sample_step)
        lane_center.append(c)
        lane_left.append(_offset_boundary(c, np.array([0.0, -1.0]), +lane_width/2))
        lane_right.append(_offset_boundary(c, np.array([0.0, -1.0]), -lane_width/2))

        # 남->서: 중심 (-intersection_half, -intersection_half)
        c = _make_arc(center=np.array([-intersection_half, -intersection_half]),
                      r=r, th0=+1.5*np.pi, th1=+1.0*np.pi, step_m=sample_step)
        lane_center.append(c)
        lane_left.append(_offset_boundary(c, np.array([-1.0, 0.0]), +lane_width/2))
        lane_right.append(_offset_boundary(c, np.array([-1.0, 0.0]), -lane_width/2))

    # ---------------------------------------
    # 5) 정지선 (stop line)
    # ---------------------------------------
    # y=0에서 y=-7까지의 세로선 (x 좌표는 적절한 위치에 설정)
    stop_x = -5.0  # x 좌표 (필요시 조정 가능)
    stop_line.append(np.array([
        [stop_x, 0.0, z0],
        [stop_x, -7.0, z0]
    ]))

    # ---------------------------------------
    # 6) 선행차(예: 동쪽에서 서쪽으로 가는 차 1대)
    # ---------------------------------------
    # 교차로 중심 근처에 배치하여 테스트하기 쉽게
    L, W, H = 4.5, 1.6, 1.5
    lane_center_offset = 0.5 * lane_width
    car_center_y = -lane_center_offset + (lane_width - W) / 2
    car_center = np.array([+15.0, car_center_y, z0], dtype=float)
    box = []
    for dz in [0.0, H]:
        for dx in [-L/2, L/2]:
            for dy in [-W/2, W/2]:
                box.append(car_center + np.array([dx, dy, dz], dtype=float))
    car_box = np.array(box, dtype=float)

    return {
        "lane_center": lane_center,  # List[np.ndarray]
        "lane_left": lane_left,
        "lane_right": lane_right,
        "car_box": car_box,
        "stop_line": stop_line,
    }