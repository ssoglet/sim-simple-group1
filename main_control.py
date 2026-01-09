from __future__ import annotations

import numpy as np

# from bev
from src.sensors.bev import world_xy_to_bev, clip_bev

# from map
from src.world.world import make_demo_world_points, make_pangyo_world_pts
from src.world.osm_converter import make_osm_world_pts
from src.common.types import TrafficLightState

# from kinetic bicycle model
from src.dynamics.vehicle import VehicleState, VehicleParams, step_kinematic_bicycle

from main_viewer import show_bev_viewer

# from control
from src.dynamics.control import (
    ControlParams,
    nearest_point_on_path,
    signed_lateral_error,
    heading_error,
    pure_pursuit_steer,
    speed_control
)
from src.planning.path import (
    make_straight_lane_path, 
    make_lane_change_path,
    make_obstacle_avoidance_path,
    extract_lane_boundaries_from_world
)

# New modules
from src.perception.perception import PerceptionModule
from src.control.aeb import AEBModule, AEBParams

def main():
    # --- BEV window bounds (meters) ---
    xlim = (-50.0, 100.0)
    ylim = (-20.0, 20.0)
    bev_size = (900, 600)

    # --- World (맵 선택) ---
    # "demo": 기본 데모 맵
    # "pangyo": 판교 교차로 맵
    # "osm": OSM 맵 파일 기반
    map_type = "pangyo"  # "demo", "pangyo", "osm" 중 선택
    
    if map_type == "osm":
        # OSM 맵 사용
        world_pts = make_osm_world_pts()
        # OSM 맵의 경우 범위 자동 계산 또는 넓은 범위 설정
        xlim = (-500.0, 500.0)
        ylim = (-500.0, 500.0)
    elif map_type == "pangyo":
        # 판교 교차로 맵
        world_pts = make_pangyo_world_pts(
            area_half=200.0,
            intersection_half=25.0,
            lane_width=3.5,
            lanes_each_dir=2
        )
        # 교차로 중심 주변 확대하여 장애물 회피 경로를 자세히 보기
        xlim = (-50.0, 50.0)
        ylim = (-30.0, 30.0)
    else:
        # 기본 데모 맵
        world_pts = make_demo_world_points()
        xlim = (-50.0, 100.0)
        ylim = (-20.0, 20.0)

    bev_dict = {}
    for name, pts3 in world_pts.items():
        if pts3 is None:
            continue
        
        # 판교 맵의 경우 수직 차선(위/아래 방향) 필터링
        filter_vertical = (map_type == "pangyo" and name in ("lane_center", "lane_left", "lane_right"))
            
        # OSM 맵의 경우 리스트 형태일 수 있음
        if isinstance(pts3, list):
            # 리스트의 각 polyline을 처리
            bev_uv_list = []
            for polyline in pts3:
                if polyline is None or len(polyline) == 0:
                    continue
                try:
                    polyline_array = np.asarray(polyline)
                    if polyline_array.ndim == 2 and polyline_array.shape[1] >= 2:
                        # 수직 차선 필터링: y 방향 변화가 큰 차선 제거
                        if filter_vertical and len(polyline_array) >= 2:
                            # 첫 점과 마지막 점의 좌표 차이
                            start_pt = polyline_array[0, :2]
                            end_pt = polyline_array[-1, :2]
                            dx = abs(end_pt[0] - start_pt[0])
                            dy = abs(end_pt[1] - start_pt[1])
                            
                            # y 방향 변화가 x 방향 변화보다 크면 수직 차선으로 판단하여 제외
                            if dy > dx:
                                continue
                        
                        bev_uv = world_xy_to_bev(polyline_array, xlim=xlim, ylim=ylim, bev_size=bev_size)
                        bev_uv = clip_bev(bev_uv, bev_size)
                        bev_uv_list.append(bev_uv)
                except Exception as e:
                    print(f"Warning: Failed to process {name} polyline: {e}")
                    continue
            # 리스트 형태로 저장 (뷰어에서 리스트를 처리할 수 있어야 함)
            bev_dict[name] = bev_uv_list if bev_uv_list else None
        else:
            # 단일 numpy array 형태
            try:
                pts3_array = np.asarray(pts3)
                if pts3_array.ndim == 2 and pts3_array.shape[1] >= 2:
                    bev_uv = world_xy_to_bev(pts3_array, xlim=xlim, ylim=ylim, bev_size=bev_size)
                    bev_uv = clip_bev(bev_uv, bev_size)
                    bev_dict[name] = bev_uv
            except Exception as e:
                print(f"Warning: Failed to process {name}: {e}")
                continue

    # --- Reference path ---
    if map_type == "osm":
        # OSM 맵: 맵의 실제 경로를 따라가도록 (임시로 직선 경로)
        # TODO: OSM 맵에서 실제 경로 추출하여 사용
        base_ref_path = make_straight_lane_path(x_start=-400.0, x_end=400.0, y=0.0)
    elif map_type == "pangyo":
        # 판교 맵: 교차로 중심 근처에서 시작하여 동쪽으로 가는 경로
        # 교차로 중심(0,0)에서 서쪽으로 25m 앞에서 시작, 동쪽으로 40m까지
        base_ref_path = make_straight_lane_path(x_start=-25.0, x_end=40.0, y=-1.75)
    else:
        # 기본 데모 맵
        base_ref_path = make_straight_lane_path(x_start=0.0, x_end=80.0, y=-1.0)
    
    ref_path = base_ref_path.copy()

    # --- Vehicle and controllers ---
    vparams = VehicleParams(wheelbase=2.8)
    cparams = ControlParams(lookahead_base=4.0, lookahead_gain=0.25, k_speed=0.8)
    
    # --- New modules ---
    perception = PerceptionModule(
        detection_range=50.0,
        safety_margin=2.0,
        vehicle_length=4.5,
        vehicle_width=2.0
    )
    
    aeb_module = AEBModule(AEBParams(
        ttc_threshold_warning=5.0,
        ttc_threshold_braking=3.0,
        ttc_threshold_emergency=1.5,
        min_distance_warning=15.0,
        min_distance_braking=10.0,
        min_distance_emergency=5.0
    ))

    # 초기 상태 설정
    if map_type == "osm":
        state = VehicleState(x=-400.0, y=0.0, yaw=np.deg2rad(0.0), v=0.0)
    elif map_type == "pangyo":
        # 교차로 중심(0,0) 근처에서 시작: 서쪽으로 25m 앞, 오른쪽 차선
        state = VehicleState(x=-25.0, y=-1.75, yaw=np.deg2rad(0.0), v=0.0)
    else:
        state = VehicleState(x=0.0, y=-1.0, yaw=np.deg2rad(0.0), v=0.0)
    
    v_ref = 9.0 

    # --- Simulation loop ---
    dt = 0.05
    T = 20.0
    steps = int(T / dt)

    states = [state]
    traj = [(state.x, state.y)]
    obstacle_detections_log = []
    aeb_activations = []
    path_replan_count = 0
    collision_warnings = []
    
    # 신호등 상태 로그 (각 프레임별)
    traffic_light_states_log = []

    for k in range(steps):
        # --- Perception: 장애물 감지 ---
        obstacles = perception.detect_obstacles(state, world_pts, ref_path)
        obstacle_detections_log.append(obstacles)
        
        # --- Collision avoidance: 장애물 회피 경로 생성 ---
        critical_obstacle = None
        for obs in obstacles:
            # 장애물이 경로상에 있거나 충돌 위험이 있는 경우
            # 거리 조건도 함께 확인하여 더 민감하게 감지
            if obs.is_in_path or obs.collision_risk in ['medium', 'high', 'critical']:
                # 앞쪽에 있고 충분히 가까운 장애물만 고려
                if obs.distance < 50.0:  # 감지 범위 확대
                    if critical_obstacle is None or obs.distance < critical_obstacle.distance:
                        critical_obstacle = obs
        
        # 디버깅: 장애물 감지 정보
        if obstacles and k % 20 == 0:  # 20프레임마다 출력
            for obs in obstacles:
                if obs.is_in_path or obs.distance < 50.0:
                    collision_warnings.append((k * dt, obs.distance, obs.is_in_path, obs.collision_risk))
        
        # 위험한 장애물이 있고 회피가 필요한 경우 경로 재계획
        # 거리 조건을 완화하여 더 일찍 회피 시작
        if critical_obstacle is not None and critical_obstacle.distance < 50.0:
            # 차선 경계 추출
            lane_boundaries = extract_lane_boundaries_from_world(
                world_pts, state, lane_width=3.5
            )
            
            # 차선 경계가 없으면 기본값 사용
            if lane_boundaries is None:
                current_y = state.y
                if current_y < 0:
                    lane_boundaries = (-3.5, 0.0)
                else:
                    lane_boundaries = (0.0, 3.5)
            
            ref_path = make_obstacle_avoidance_path(
                state,
                base_ref_path,
                critical_obstacle.position,
                critical_obstacle.size,
                avoidance_offset=2.5,
                lane_boundaries=lane_boundaries
            )
            path_replan_count += 1
        else:
            # 안전한 경우 원래 경로 사용
            # 하지만 차선 제약은 항상 적용
            lane_boundaries = extract_lane_boundaries_from_world(
                world_pts, state, lane_width=3.5
            )
            
            if lane_boundaries is None:
                current_y = state.y
                if current_y < 0:
                    lane_boundaries = (-3.5, 0.0)
                else:
                    lane_boundaries = (0.0, 3.5)
            
            # 차선 내에서만 경로 사용
            ref_path = base_ref_path.copy()
            vehicle_width_half = 1.0
            lane_min_y, lane_max_y = lane_boundaries
            
            # 경로의 모든 점이 차선 내에 있도록 제한
            for i in range(len(ref_path)):
                ref_path[i, 1] = np.clip(
                    ref_path[i, 1],
                    lane_min_y + vehicle_width_half,
                    lane_max_y - vehicle_width_half
                )
        
        # --- 신호등 상태 확인 (차량 제어 전에) ---
        frame_in_cycle = k % 200
        if frame_in_cycle < 120:
            tl_state = TrafficLightState.RED
        else:
            tl_state = TrafficLightState.GREEN
        traffic_light_states_log.append({"tl_front": tl_state})
        
        # --- Control: 조향 및 속도 제어 ---
        idx_near, _, _ = nearest_point_on_path(ref_path, state.x, state.y)
        e_lat = signed_lateral_error(state, ref_path, idx_near)
        e_head = heading_error(state, ref_path, idx_near)
        
        delta, dbg = pure_pursuit_steer(state, ref_path, vparams, cparams)
        
        # --- 신호등에 따른 속도 제어 ---
        stop_line_x = -5.0
        stop_distance = 8.0 
        stop_point_x = stop_line_x - stop_distance  
        
        # 빨간불일 때
        should_stop = False
        if tl_state == TrafficLightState.RED:
            if state.x < stop_point_x:
                # 계속 주행
                v_ref_traffic = v_ref
                should_stop = False
            elif state.x < stop_line_x:
                # 멈춤
                v_ref_traffic = 0.0
                should_stop = True
            else:
                v_ref_traffic = 0.0
                should_stop = True
        else:  # GREEN
            # 초록불이면 항상 정상 주행
            v_ref_traffic = v_ref
            should_stop = False
        
        # 기본 속도 제어
        if should_stop:
            # 정지선 앞에서 강제로 강한 제동 적용
            # 현재 속도에 비례하여 강한 제동 (최대 제동력 사용)
            a_base = max(vparams.min_accel, -abs(state.v) * 5.0)  # 강한 제동
        else:
            a_base = speed_control(state.v, v_ref_traffic, vparams, cparams)
        
        # --- AEB: 자동 긴급 제동 적용 ---
        a, aeb_debug = aeb_module.compute_safe_acceleration(state, obstacles, a_base)
        if aeb_debug.get("aeb_active", False):
            aeb_activations.append((k * dt, aeb_debug))

        # --- Vehicle update ---
        state = step_kinematic_bicycle(state, delta, a, dt, vparams)
        
        # 정지선 앞에서 완전히 멈추도록 보정
        if should_stop and state.v < 0.1:  # 속도가 거의 0이면 완전히 멈춤
            state = VehicleState(x=state.x, y=state.y, yaw=state.yaw, v=0.0)
        
        states.append(state)
        traj.append((state.x, state.y))
    
    # 디버깅 정보 출력
    print(f"\n=== 시뮬레이션 완료 ===")
    print(f"경로 재계획 횟수: {path_replan_count}")
    
    if collision_warnings:
        print(f"\n=== 장애물 감지 기록 (샘플) ===")
        for t, dist, in_path, risk in collision_warnings[:10]:  # 처음 10개만
            print(f"Time: {t:.2f}s, Distance: {dist:.2f}m, InPath: {in_path}, Risk: {risk}")
    
    if aeb_activations:
        print(f"\n=== AEB 활성화 기록 ===")
        for t, info in aeb_activations:
            print(f"Time: {t:.2f}s, Reason: {info.get('reason', 'unknown')}, "
                  f"Distance: {info.get('obstacle_distance', 0):.2f}m, "
                  f"TTC: {info.get('obstacle_ttc', -1):.2f}s")
    
    # 장애물 감지 확인
    total_detections = sum(1 for obs in obstacle_detections_log if obs)
    print(f"\n장애물 감지된 프레임 수: {total_detections}/{steps}")
    
    show_bev_viewer(
        bev_dict=bev_dict,
        bev_size=bev_size,
        xlim=xlim,
        ylim=ylim,
        ref_path_xy=ref_path,
        states=states,
        traj_xy_list=traj,
        world_pts=world_pts,
        traffic_light_states_log=traffic_light_states_log,
        title="Dynamics & Control BEV Viewer"
    )
if __name__ == "__main__":
    main()
