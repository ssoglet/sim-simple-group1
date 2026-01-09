from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# from camera
from src.sensors.camera_pinhole import project_pinhole
from src.sensors.camera_extrinsic import make_T_cam_world
from src.common.types import CameraIntrinsics, TrafficLightState

# from kinetic bicycle model
from src.dynamics.vehicle import VehicleState, VehicleParams, step_kinematic_bicycle


def make_T_vehicle_world(state: VehicleState):
    """
    World(x fwd, y left, z up) -> Vehicle(x fwd, y left, z up)
    """
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    R = np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1],
    ], dtype=float)
    t = np.array([state.x, state.y, 0.0], dtype=float)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = -R @ t
    return T


def _meter_to_bev_pixel(x, y, xlim, ylim, bev_size):
    W, H = bev_size
    u = (x - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)
    v = (1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * (H - 1)
    return float(u), float(v)


def show_bev_viewer(
    bev_dict,
    bev_size,
    xlim,
    ylim,
    ref_path_xy,
    states,          # list of VehicleState
    traj_xy_list,    # list of (x,y) tuples, same length as states
    world_pts,
    title="Day2 Viewer",
    traffic_light_states_log=None,  # List[Dict[str, TrafficLightState]] - 각 프레임별 신호등 상태
):
    
    # --- helpers ---
    def _order_polygon_ccw(pts_uv: np.ndarray) -> np.ndarray:
        """Order 2D points counter-clockwise around centroid."""
        c = pts_uv.mean(axis=0)
        ang = np.arctan2(pts_uv[:,1] - c[1], pts_uv[:,0] - c[0])
        return pts_uv[np.argsort(ang)]

    W, H = bev_size
    
    # 중앙선 기준 위치 계산 (y=0)
    _, center_v = _meter_to_bev_pixel(0.0, 0.0, xlim, ylim, bev_size)

    fig = plt.figure(figsize=(16, 8), dpi=80)

    # bev panel
    ax = fig.add_axes([0.06, 0.14, 0.52, 0.82])
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # meter ticks
    xticks = np.linspace(0, W, 6)
    yticks = np.linspace(0, H, 6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.linspace(xlim[0], xlim[1], 6).astype(int))
    ax.set_yticklabels(np.linspace(ylim[1], ylim[0], 6).astype(int))
    ax.set_xlabel("x [m] (forward)")
    ax.set_ylabel("y [m] (left)")
    ax.grid(True)

    # --- 중앙선 후보 찾기: 전체 lane 중에서 center_v에 가장 가까운 polyline 하나 선택 ---
    central_key = None
    central_index = None
    best_dist = None
    for k_tmp, uv_tmp in bev_dict.items():
        if k_tmp not in ("lane_left", "lane_center", "lane_right"):
            continue
        if uv_tmp is None:
            continue
        items_tmp = uv_tmp if isinstance(uv_tmp, list) else [uv_tmp]
        for idx_poly, poly_uv in enumerate(items_tmp):
            if poly_uv is None or len(poly_uv) == 0:
                continue
            arr = np.asarray(poly_uv)
            if arr.ndim != 2 or len(arr) == 0:
                continue
            v_mean = arr[:, 1].mean()
            dist = abs(v_mean - center_v)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                central_key = k_tmp
                central_index = idx_poly

    # --- draw static map (lanes as solid lines, car_box as rectangle) ---
    for k, uv in bev_dict.items():
        if uv is None:
            continue
            
        # OSM 맵의 경우 리스트 형태일 수 있음
        if isinstance(uv, list):
            # 리스트의 각 polyline을 그리기
            if k in ("lane_left", "lane_center", "lane_right"):
                label_set = False

                for poly_idx, polyline_uv in enumerate(uv):
                    if polyline_uv is None or len(polyline_uv) == 0:
                        continue
                    try:
                        polyline_array = np.asarray(polyline_uv)
                        if polyline_array.ndim == 2 and len(polyline_array) > 0:
                            v_mean = polyline_array[:, 1].mean()
                            
                            # 이 polyline이 '가장 가운데' 중앙선인지 여부
                            is_central = (k == central_key and poly_idx == central_index)
                            
                            # 색상 결정: 중앙선은 검정, 그 위는 노란색, 아래는 하늘색
                            if is_central:
                                col = "black"
                                label_name = "lane_center"
                            elif v_mean < center_v:
                                # 중앙선 위쪽 (화면 상단)
                                col = "gold"
                                label_name = "lane_left"
                            else:
                                # 중앙선 아래쪽 (화면 하단)
                                col = "deepskyblue"
                                label_name = "lane_right"
                            
                            # 범례 등록
                            if is_central:
                                # 중앙선은 검정색인 경우에만 범례 등록
                                if not label_set:
                                    ax.plot(polyline_array[:, 0], polyline_array[:, 1], 
                                           linewidth=2.5, label=label_name, color=col)
                                    label_set = True
                                else:
                                    ax.plot(polyline_array[:, 0], polyline_array[:, 1], 
                                           linewidth=2.5, color=col)
                            else:
                                # lane_left, lane_right는 각각 한 번만 범례 등록
                                if not label_set:
                                    ax.plot(polyline_array[:, 0], polyline_array[:, 1], 
                                           linewidth=2.5, label=label_name, color=col)
                                    label_set = True
                                else:
                                    ax.plot(polyline_array[:, 0], polyline_array[:, 1], 
                                           linewidth=2.5, color=col)
                    except Exception:
                        continue
            elif k == "stop_line":
                # stop_line은 빨간 점선으로 그리기
                label_set = False
                for item_uv in uv:
                    if item_uv is None or len(item_uv) == 0:
                        continue
                    try:
                        item_array = np.asarray(item_uv)
                        if item_array.ndim == 2 and len(item_array) > 0:
                            if not label_set:
                                ax.plot(item_array[:, 0], item_array[:, 1], 
                                       linewidth=3.0, color="red", linestyle="--", label=k)
                                label_set = True
                            else:
                                ax.plot(item_array[:, 0], item_array[:, 1], 
                                       linewidth=3.0, color="red", linestyle="--")
                    except Exception:
                        continue
            else:
                # 다른 요소들도 리스트로 처리
                for item_uv in uv:
                    if item_uv is None or len(item_uv) == 0:
                        continue
                    try:
                        item_array = np.asarray(item_uv)
                        if item_array.ndim == 2 and len(item_array) > 0:
                            ax.scatter(item_array[:, 0], item_array[:, 1], s=3, label=k)
                            break  # 한 번만 라벨 표시
                    except Exception:
                        continue
        elif k in ("lane_left", "lane_center", "lane_right"):
            # solid line (단일 배열)
            uv_array = np.asarray(uv)
            if uv_array.ndim == 2 and len(uv_array) > 0:
                v_mean = uv_array[:, 1].mean()
                
                # 이 polyline이 '가장 가운데' 중앙선인지 여부
                is_central = (k == central_key and 0 == central_index)
                
                # 색상 결정: 중앙선은 검정, 그 위는 노란색, 아래는 하늘색
                if is_central:
                    col = "black"
                    label_name = "lane_center"
                elif v_mean < center_v:
                    # 중앙선 위쪽 (화면 상단)
                    col = "gold"
                    label_name = "lane_left"
                else:
                    # 중앙선 아래쪽 (화면 하단)
                    col = "deepskyblue"
                    label_name = "lane_right"
                
                ax.plot(uv_array[:, 0], uv_array[:, 1], linewidth=2.5, label=label_name, color=col)
        elif k == "car_box":
            # car_box points are corners of ONE car (not 4 cars)
            # If 8 points exist (bottom+top), take bottom 4 (first 4 in our demo generation)
            car_uv = np.asarray(uv)
            if car_uv.ndim == 2 and car_uv.shape[0] >= 4:
                bottom4 = car_uv[:4].copy()  # assumes first 4 are dz=0
                bottom4 = _order_polygon_ccw(bottom4)
                poly = np.vstack([bottom4, bottom4[0]])  # close loop
                ax.plot(poly[:, 0], poly[:, 1], linewidth=2.5, color="lime", label="car_box(rect)")
            elif car_uv.ndim == 2:
                ax.scatter(car_uv[:, 0], car_uv[:, 1], s=30, label="car_box")
        elif k == "stop_line":
            # stop_line은 빨간 점선으로 그리기
            uv_array = np.asarray(uv)
            if uv_array.ndim == 2 and len(uv_array) > 0:
                ax.plot(uv_array[:, 0], uv_array[:, 1], 
                       linewidth=3.0, color="red", linestyle="--", label=k)
        else:
            # 기타 요소들
            uv_array = np.asarray(uv)
            if uv_array.ndim == 2 and len(uv_array) > 0:
                ax.scatter(uv_array[:, 0], uv_array[:, 1], s=3, label=k)

    # draw static reference path (lane change)
    lane_width = 3.5

    right_lane_1_y = -lane_width * 0.5   # 도착
    right_lane_2_y = -lane_width * 1.5   # 시작

    # x 기준 보간 범위
    x_start = ref_path_xy[0][0]
    x_end   = ref_path_xy[-1][0]

    ref_uv = []
    for (x, _) in ref_path_xy:
        # x 위치에 따른 0~1 보간 계수
        t = (x - x_start) / (x_end - x_start + 1e-6)
        
        # y를 선형으로 이동 (2차선 → 1차선)
        y = (1 - t) * right_lane_2_y + t * right_lane_1_y
        
        u, v = _meter_to_bev_pixel(x, y, xlim, ylim, bev_size)
        ref_uv.append((u, v))

    ref_uv = np.array(ref_uv, dtype=float)
    ax.plot(ref_uv[:, 0], ref_uv[:, 1], linewidth=2, label="reference path")

    # dynamic artists (will update)
    traj_scatter = ax.scatter([], [], s=20, facecolors='none', edgecolors='red', label="trajectory")
    ego_pt = ax.scatter([], [], s=90, marker="+")
    heading_line, = ax.plot([], [], linewidth=2)
    ax.legend(loc="upper right")
    
    # 신호등 표시용 (동적 업데이트)
    traffic_light_artists = {"circle": None, "text": None}
    
    # 초기 차량 위치 기준으로 신호등 위치 고정 계산
    # 정지선이 x=0에 있으므로, 신호등을 정지선보다 오른쪽(x > 0)에 배치
    initial_state = states[0] if states else None
    if initial_state is not None:
        # 정지선보다 오른쪽에 배치 (x > 0)
        # 정지선이 x=0이므로, 신호등을 x=5.0 정도에 배치
        fixed_tl_x = 0.0  # 정지선(x=0)보다 오른쪽
        fixed_tl_y = -7.5  # y 좌표 고정
        fixed_tl_z = 5.0  # 높이
    else:
        fixed_tl_x = 5.0
        fixed_tl_y = -7.5
        fixed_tl_z = 5.0

    # cam panel
    ax_cam = fig.add_axes([0.62, 0.14, 0.36, 0.82])

    K = CameraIntrinsics(
        fx=400.0, fy=400.0,
        cx=640.0, cy=360.0,
        width=1280, height=720
    )

    ax_cam.set_title("Pinhole Camera Sensor Viewer")
    ax_cam.set_xlim(0, K.width)
    ax_cam.set_ylim(K.height, 0)
    ax_cam.set_xlabel("u [px]")
    ax_cam.set_ylabel("v [px]")
    ax_cam.grid(True)

    cam_artists = []  # frame마다 지우고 다시 그림

    # state for viewer
    idx = {"i": 0}
    playing = {"on": False}
    timer = fig.canvas.new_timer(interval=50)  # ms

    def draw_frame(i):
        i = int(np.clip(i, 0, len(states) - 1))
        idx["i"] = i
        s = states[i] 

        # 1. 궤적(Trajectory) 업데이트: ty에 -1을 곱하여 y=0 대칭 반전
        traj_uv = []
        for tx, ty in traj_xy_list[:i+1]:
            # ty -> -ty로 변경하여 상하 반전
            u, v = _meter_to_bev_pixel(tx, -ty, xlim, ylim, bev_size)
            traj_uv.append([u, v])
        
        if len(traj_uv) > 0:
            traj_scatter.set_offsets(np.array(traj_uv))
        
        # 2. 에고 차량(Ego) 위치 업데이트: s.y에 -1을 곱하여 반전
        # 차량의 실제 y가 아닌 반전된 위치(-s.y)를 시각화
        eu, ev = _meter_to_bev_pixel(s.x, -s.y, xlim, ylim, bev_size)
        ego_pt.set_offsets(np.array([[eu, ev]]))

        # 3. 헤딩 라인(Heading Line) 업데이트
        # 위치(-s.y)와 방향(-s.yaw) 모두 반전시켜야 헤딩 라인이 경로를 따라갑니다.
        Lh = 5.0
        rev_y = -s.y
        rev_yaw = -s.yaw
        
        hx = s.x + Lh * np.cos(rev_yaw)
        hy = rev_y + Lh * np.sin(rev_yaw) 
        hu, hv = _meter_to_bev_pixel(hx, hy, xlim, ylim, bev_size)
        heading_line.set_data([eu, hu], [ev, hv])

        # 4. 신호등 표시 로직 (기존 변수 traffic_light_artists 유지)
        if traffic_light_artists["circle"] is not None:
            traffic_light_artists["circle"].remove()
        if traffic_light_artists["text"] is not None:
            traffic_light_artists["text"].remove()
        
        tl_u, tl_v = _meter_to_bev_pixel(fixed_tl_x, fixed_tl_y, xlim, ylim, bev_size)
        
        if traffic_light_states_log and i < len(traffic_light_states_log):
            current_state = traffic_light_states_log[i].get("tl_front", TrafficLightState.RED)
        else:
            current_state = TrafficLightState.RED
        
        tl_col = "red" if current_state == TrafficLightState.RED else "green"
        
        traffic_light_artists["circle"] = plt.Circle((tl_u, tl_v), 5, facecolor=tl_col, 
                                                     edgecolor="black", linewidth=2, zorder=10)
        ax.add_patch(traffic_light_artists["circle"])
        
        traffic_light_artists["text"] = ax.text(tl_u, tl_v + 15, current_state.value.upper(), 
                                                fontsize=10, ha="center", va="top", 
                                                color=tl_col, weight="bold")

        # 5. 카메라 뷰 업데이트 (이 부분에서 에러가 나면 Play가 멈춤)
        for a in cam_artists:
            try: a.remove()
            except: pass
        cam_artists.clear()

        # 실제 차량 상태 s를 사용하여 카메라 행렬 생성
        T_cam_world = make_T_cam_world() @ make_T_vehicle_world(s)
        
        # (이후 기존의 world_pts 순회 및 project_pinhole 로직...)
        
        # 신호등 3D 위치를 카메라로 프로젝션
        tl_3d = np.array([[fixed_tl_x, fixed_tl_y, fixed_tl_z]], dtype=float)
        tl_uv, visible = project_pinhole(tl_3d, T_cam_world, K)
        
        if visible and tl_uv.shape[0] > 0:
            tl_cam_u, tl_cam_v = tl_uv[0, 0], tl_uv[0, 1]
            if 0 <= tl_cam_u < K.width and 0 <= tl_cam_v < K.height:
                cam_color = "red" if current_state == TrafficLightState.RED else "green"
                
                # color= 대신 facecolor= 사용
                circle_tl = plt.Circle((tl_cam_u, tl_cam_v), 8, 
                                      facecolor=cam_color, 
                                      edgecolor="black", 
                                      linewidth=2, zorder=10)
                ax_cam.add_patch(circle_tl)
                cam_artists.append(circle_tl)
                
                # 카메라 뷰에 신호등 텍스트 표시
                text_tl = ax_cam.text(tl_cam_u, tl_cam_v + 12, 
                                      current_state.value.upper(), 
                                      fontsize=8, ha="center", va="top", 
                                      color=cam_color, weight="bold")
                cam_artists.append(text_tl)

        # 카메라 뷰에서도 중앙선 찾기 (y=0에 가장 가까운 것)
        central_cam_key = None
        central_cam_index = None
        best_cam_dist = None
        for name_tmp, pts3_tmp in world_pts.items():
            if name_tmp not in ("lane_left", "lane_center", "lane_right"):
                continue
            if pts3_tmp is None:
                continue
            items_tmp = pts3_tmp if isinstance(pts3_tmp, list) else [pts3_tmp]
            for idx_poly, poly_tmp in enumerate(items_tmp):
                if poly_tmp is None or len(poly_tmp) == 0:
                    continue
                arr = np.asarray(poly_tmp)
                if arr.ndim != 2 or len(arr) == 0:
                    continue
                y_mean = arr[:, 1].mean()  # world y 좌표
                dist = abs(y_mean - 0.0)  # y=0에 가까운지
                if best_cam_dist is None or dist < best_cam_dist:
                    best_cam_dist = dist
                    central_cam_key = name_tmp
                    central_cam_index = idx_poly

        for name, pts3 in world_pts.items():
            if pts3 is None:
                continue
                
            # OSM 맵의 경우 리스트 형태일 수 있음
            if isinstance(pts3, list):
                # 리스트의 각 polyline을 개별적으로 처리
                label_set_center = False
                label_set_left = False
                label_set_right = False
                for poly_idx, polyline in enumerate(pts3):
                    if polyline is None or len(polyline) == 0:
                        continue
                    try:
                        polyline_array = np.asarray(polyline)
                        if polyline_array.ndim == 2 and len(polyline_array) > 0:
                            uv, _ = project_pinhole(polyline_array, T_cam_world, K)
                            if uv.shape[0] == 0:
                                continue
                            
                            y_mean = polyline_array[:, 1].mean()  # world y 좌표
                            
                            # 이 polyline이 '가장 가운데' 중앙선인지 여부
                            is_central = (name == central_cam_key and poly_idx == central_cam_index)
                            
                            # 색상 결정: 중앙선은 검정, 그 위는 노란색, 아래는 하늘색
                            if is_central:
                                col = "black"
                                label_name = "lane_center"
                                use_label = not label_set_center
                                label_set_center = True
                            elif y_mean > 0:
                                # 중앙선 위쪽 (y > 0, 화면 상단)
                                col = "gold"
                                label_name = "lane_left"
                                use_label = not label_set_left
                                label_set_left = True
                            else:
                                # 중앙선 아래쪽 (y < 0, 화면 하단)
                                col = "deepskyblue"
                                label_name = "lane_right"
                                use_label = not label_set_right
                                label_set_right = True
                            
                            if name == "stop_line":
                                col = "red"
                                label_name = "stop_line"
                                use_label = True
                            elif name == "car_box":
                                col = "lime"
                                label_name = "car_box"
                                use_label = True
                            
                            # 범례 등록
                            if use_label and name in ("lane_left", "lane_center", "lane_right", "stop_line", "car_box"):
                                sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col, label=label_name)
                            else:
                                sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col)
                            cam_artists.append(sc)
                    except Exception:
                        continue
            else:
                # 단일 numpy array 형태
                try:
                    pts3_array = np.asarray(pts3)
                    if pts3_array.ndim == 2 and len(pts3_array) > 0:
                        uv, _ = project_pinhole(pts3_array, T_cam_world, K)
                        if uv.shape[0] == 0:
                            continue
                        
                        y_mean = pts3_array[:, 1].mean()  # world y 좌표
                        
                        # 이 polyline이 '가장 가운데' 중앙선인지 여부
                        is_central = (name == central_cam_key and 0 == central_cam_index)
                        
                        # 색상 결정: 중앙선은 검정, 그 위는 노란색, 아래는 하늘색
                        if is_central:
                            col = "black"
                            label_name = "lane_center"
                        elif y_mean > 0:
                            # 중앙선 위쪽 (y > 0)
                            col = "gold"
                            label_name = "lane_left"
                        else:
                            # 중앙선 아래쪽 (y < 0)
                            col = "deepskyblue"
                            label_name = "lane_right"
                        
                        if name == "stop_line":
                            col = "red"
                            label_name = "stop_line"
                        elif name == "car_box":
                            col = "lime"
                            label_name = "car_box"
                        elif name not in ("lane_left", "lane_center", "lane_right"):
                            col = "gray"
                            label_name = name
                        
                        sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col, label=label_name)
                        cam_artists.append(sc)
                except Exception:
                    continue

        # legend 중복 방지: 매 프레임 새로 갱신
        if len(cam_artists) > 0:
            ax_cam.legend(loc="upper right")

        ax.set_title(f"{title}  |  frame {i+1}/{len(states)}  |  v={s.v:.2f} m/s  yaw={np.rad2deg(s.yaw):.1f}°")
        fig.canvas.draw_idle()

    def on_prev(event):
        playing["on"] = False
        draw_frame(idx["i"] - 1)

    def on_next(event):
        playing["on"] = False
        draw_frame(idx["i"] + 1)

    def on_play_pause(event):
        playing["on"] = not playing["on"]

    def _tick():
        if not playing["on"]:
            return
        if idx["i"] >= len(states) - 1:
            playing["on"] = False
            return
        draw_frame(idx["i"] + 1)

    timer.add_callback(_tick)
    timer.start()

    # Buttons
    ax_prev = fig.add_axes([0.20, 0.03, 0.10, 0.07])
    ax_next = fig.add_axes([0.32, 0.03, 0.10, 0.07])
    ax_play = fig.add_axes([0.44, 0.03, 0.12, 0.07])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    b_play = Button(ax_play, "Play/Pause")

    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_play.on_clicked(on_play_pause)

    # initial
    draw_frame(0)
    plt.show()