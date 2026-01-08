from __future__ import annotations
import numpy as np

def make_straight_lane_path(x_start: float = 0.0, x_end: float = 80.0, y: float = 0.0, num: int = 400) -> np.ndarray:
    xs = np.linspace(x_start, x_end, num)
    ys = np.full_like(xs, y, dtype=float)
    return np.stack([xs, ys], axis=1)

def make_lane_change_path(x_start: float = 0.0, x_end: float = 80.0,
                          y0: float = 0.0, y1: float = 3.5,
                          x_change_start: float = 20.0, x_change_end: float = 40.0,
                          num: int = 500) -> np.ndarray:
    """
    Simple smooth lane change using cubic interpolation on y(x).
    """
    xs = np.linspace(x_start, x_end, num)
    ys = np.zeros_like(xs, dtype=float)

    for i, x in enumerate(xs):
        if x < x_change_start:
            ys[i] = y0
        elif x > x_change_end:
            ys[i] = y1
        else:
            # normalize s in [0,1]
            s = (x - x_change_start) / (x_change_end - x_change_start)
            # smoothstep cubic: 3s^2 - 2s^3
            w = 3*s*s - 2*s*s*s
            ys[i] = (1-w)*y0 + w*y1

    return np.stack([xs, ys], axis=1)

def make_turn_path(center, radius, angle_start, angle_end, num=100):
    angles = np.linspace(angle_start, angle_end, num)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return list(zip(xs, ys))

def extract_lane_boundaries_from_world(
    world_pts: dict,
    ego_state,
    lane_width: float = 3.5
) -> tuple[float, float] | None:
    """
    새로운 맵 구조에서 현재 차량 위치 기준으로 차선 경계 추출
    OSM 맵, 판교 맵, 데모 맵 모두 지원
    
    Args:
        world_pts: 월드 포인트 딕셔너리 (lane_center, lane_left, lane_right 포함)
        ego_state: 자차 상태
        lane_width: 차선 폭 (m)
    
    Returns:
        (min_y, max_y) 차선 경계 또는 None
    """
    if "lane_center" not in world_pts:
        return None
    
    lane_centers = world_pts["lane_center"]
    ego_pos = np.array([ego_state.x, ego_state.y])
    
    # OSM 맵의 경우: lane_left/lane_right가 None이므로 차선 중심만 사용
    if isinstance(lane_centers, list):
        # List 형태의 차선들 중에서 가장 가까운 차선 찾기
        min_dist = float('inf')
        closest_lane_y = None
        closest_lane_heading = None
        
        for lane_center in lane_centers:
            if lane_center is None or len(lane_center) == 0:
                continue
            lane_2d = lane_center[:, :2]  # (N, 2)
            
            # 차선의 각 점에서 거리 계산
            for i in range(len(lane_2d)):
                point = lane_2d[i]
                dist = np.linalg.norm(point - ego_pos)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_lane_y = point[1]  # y 좌표
                    
                    # 가장 가까운 점 근처에서 헤딩 계산
                    if i < len(lane_2d) - 1:
                        next_point = lane_2d[i + 1]
                        dx = next_point[0] - point[0]
                        dy = next_point[1] - point[1]
                        closest_lane_heading = np.arctan2(dy, dx)
        
        if closest_lane_y is not None:
            # OSM 맵의 경우: 차선 중심 기준으로 경계 설정
            # 차선이 수평인지 수직인지에 따라 다르게 처리
            if closest_lane_heading is not None:
                # 수직 차선인지 확인 (헤딩이 90도 또는 -90도 근처)
                heading_deg = np.rad2deg(closest_lane_heading)
                is_vertical = abs(abs(heading_deg) - 90) < 30
                
                if is_vertical:
                    # 수직 차선: x 방향으로 제한
                    if closest_lane_y < 0:
                        return (closest_lane_y - lane_width/2, 0.0)
                    else:
                        return (0.0, closest_lane_y + lane_width/2)
                else:
                    # 수평 차선: y 방향으로 제한
                    if closest_lane_y < 0:
                        return (-3.5, 0.0)
                    else:
                        return (0.0, 3.5)
            else:
                # 기본값: 중앙선 기준
                if closest_lane_y < 0:
                    return (-3.5, 0.0)
                else:
                    return (0.0, 3.5)
    
    return None

def make_obstacle_avoidance_path(
    ego_state,
    ref_path: np.ndarray,
    obstacle_pos: np.ndarray,
    obstacle_size: tuple[float, float],
    avoidance_offset: float = 2.0,
    lane_boundaries: tuple[float, float] = None,
    num: int = 400
) -> np.ndarray:
    """
    장애물을 회피하는 경로 생성 (차선 제약 포함)
    
    Args:
        ego_state: 자차 상태
        ref_path: 원래 참조 경로 (N, 2)
        obstacle_pos: 장애물 위치 [x, y]
        obstacle_size: 장애물 크기 (length, width)
        avoidance_offset: 회피 시 추가 여유 (m)
        lane_boundaries: 차선 경계 (min_y, max_y) - 중앙선을 넘지 않도록 제한
        num: 경로 점 개수
    
    Returns:
        회피 경로 (N, 2)
    """
    if len(ref_path) < 2:
        return ref_path
    
    # 차선 경계 설정 (기본값: 중앙선(y=0)을 넘지 않음)
    if lane_boundaries is None:
        # 현재 차량 위치 기준으로 차선 결정
        current_y = ego_state.y
        if current_y < 0:
            # 오른쪽 차선: 중앙선(0)을 넘지 않음
            lane_boundaries = (-3.5, 0.0)
        else:
            # 왼쪽 차선: 중앙선(0)을 넘지 않음
            lane_boundaries = (0.0, 3.5)
    
    lane_min_y, lane_max_y = lane_boundaries
    vehicle_width_half = 1.0  # 차량 폭의 절반 (안전 여유 포함)
    
    # 장애물 회피 경로 생성
    x_start = ego_state.x
    x_end = ref_path[-1, 0]
    
    # 장애물의 y 위치 확인하여 좌우 회피 결정
    ref_y = ref_path[0, 1]
    obs_y = obstacle_pos[1]
    
    # 장애물의 y 범위 계산
    obs_y_min = obs_y - obstacle_size[1] / 2
    obs_y_max = obs_y + obstacle_size[1] / 2
    
    # 차선 내에서 회피 공간 확인
    available_space_below = ref_y - vehicle_width_half - lane_min_y
    available_space_above = lane_max_y - (ref_y + vehicle_width_half)
    
    # 우측 회피 시도 (더 아래로)
    avoidance_y_right = ref_y - (obstacle_size[1] / 2 + avoidance_offset)
    # 좌측 회피 시도 (더 위로)
    avoidance_y_left = ref_y + (obstacle_size[1] / 2 + avoidance_offset)
    
    # 차선 제약 확인하여 가능한 회피 방향 선택
    avoidance_y = None
    
    # 우측 회피가 가능한지 확인 (차선 경계 내)
    if avoidance_y_right - vehicle_width_half >= lane_min_y:
        avoidance_y = avoidance_y_right
    # 좌측 회피가 가능한지 확인
    elif avoidance_y_left + vehicle_width_half <= lane_max_y:
        avoidance_y = avoidance_y_left
    else:
        # 양쪽 모두 불가능하면 현재 차선 내에서 최대한 회피
        if available_space_below > available_space_above:
            avoidance_y = max(lane_min_y + vehicle_width_half, 
                            ref_y - (obstacle_size[1] / 2 + avoidance_offset / 3))
        else:
            avoidance_y = min(lane_max_y - vehicle_width_half,
                            ref_y + (obstacle_size[1] / 2 + avoidance_offset / 3))
    
    # 차선 경계 내로 제한
    avoidance_y = np.clip(avoidance_y, lane_min_y + vehicle_width_half, 
                         lane_max_y - vehicle_width_half)
    
    # 장애물 회피 시작/종료 지점 계산
    obs_x_start = obstacle_pos[0] - obstacle_size[0] / 2 - 5.0
    obs_x_end = obstacle_pos[0] + obstacle_size[0] / 2 + 5.0
    
    # 경로 생성
    xs = np.linspace(x_start, x_end, num)
    ys = np.zeros_like(xs, dtype=float)
    
    for i, x in enumerate(xs):
        if x < obs_x_start:
            # 장애물 이전: 원래 경로 유지
            t = (x - x_start) / (x_end - x_start) if x_end != x_start else 0
            idx = int(t * (len(ref_path) - 1))
            idx = np.clip(idx, 0, len(ref_path) - 1)
            ys[i] = ref_path[idx, 1]
            # 차선 제약 적용
            ys[i] = np.clip(ys[i], lane_min_y + vehicle_width_half,
                           lane_max_y - vehicle_width_half)
        elif x > obs_x_end:
            # 장애물 이후: 원래 경로로 복귀
            t = (x - x_start) / (x_end - x_start) if x_end != x_start else 1
            idx = int(t * (len(ref_path) - 1))
            idx = np.clip(idx, 0, len(ref_path) - 1)
            ys[i] = ref_path[idx, 1]
            # 차선 제약 적용
            ys[i] = np.clip(ys[i], lane_min_y + vehicle_width_half,
                           lane_max_y - vehicle_width_half)
        else:
            # 장애물 구간: 회피 경로
            s = (x - obs_x_start) / (obs_x_end - obs_x_start)
            s = np.clip(s, 0, 1)
            
            y_before = ref_y
            y_after = ref_y
            
            # 부드러운 곡선으로 연결
            if s < 0.5:
                s_norm = s * 2.0
                w = 3 * s_norm**2 - 2 * s_norm**3  # smoothstep
                ys[i] = (1 - w) * y_before + w * avoidance_y
            else:
                s_norm = (s - 0.5) * 2.0
                w = 3 * s_norm**2 - 2 * s_norm**3  # smoothstep
                ys[i] = (1 - w) * avoidance_y + w * y_after
            
            # 차선 제약 적용
            ys[i] = np.clip(ys[i], lane_min_y + vehicle_width_half,
                           lane_max_y - vehicle_width_half)
    
    return np.stack([xs, ys], axis=1)
