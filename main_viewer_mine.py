from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# from camera
from src.sensors.camera_pinhole import project_pinhole
from src.sensors.camera_extrinsic import make_T_cam_world
from src.common.types import CameraIntrinsics

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
    bev_dict, bev_size, xlim, ylim, ref_path_xy,
    states, traj_xy_list, world_pts, title="Day2 Viewer"
):
    # --- Helpers ---
    def _order_polygon_ccw(pts_uv: np.ndarray) -> np.ndarray:
        c = pts_uv.mean(axis=0)
        ang = np.arctan2(pts_uv[:,1] - c[1], pts_uv[:,0] - c[0])
        return pts_uv[np.argsort(ang)]

    W, H = bev_size
    fig = plt.figure(figsize=(16, 8), dpi=80)

    # 1. BEV Panel 설정
    ax = fig.add_axes([0.06, 0.14, 0.52, 0.82])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    xticks = np.linspace(0, W, 6)
    yticks = np.linspace(0, H, 6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.linspace(xlim[0], xlim[1], 6).astype(int))
    ax.set_yticklabels(np.linspace(ylim[1], ylim[0], 6).astype(int))
    ax.set_xlabel("x [m] (forward)")
    ax.set_ylabel("y [m] (left)")
    ax.grid(True)

    _, center_v = _meter_to_bev_pixel(0.0, 0.0, xlim, ylim, bev_size)

    # 2. Static Map (BEV)
    for k, uv in bev_dict.items():
        if uv is None: continue
        items = uv if isinstance(uv, list) else [uv]
        
        # 해당 카테고리(예: lane_center)에서 라벨이 한 번만 등록되게 관리
        label_set = False

        for item in items:
            pts = np.asarray(item)
            if pts.ndim != 2 or len(pts) == 0: continue

            # 색상 결정
            if k == "lane_left":
                col = "gold"
            elif k == "lane_right":
                col = "deepskyblue"
            elif k == "lane_center":
                # [핵심 수정] 픽셀 위치 v를 보고 중앙선(검정)인지 판단
                v_mean = pts[:, 1].mean()
                if abs(v_mean - center_v) < 2.0:
                    col = "black"  # 중앙선
                else:
                    # 중앙선이 아닌 lane_center 소속 차선들 (파랑/금색)
                    col = "gold" if v_mean < center_v else "deepskyblue"
            else:
                col = "gray"

            # --- 범례(Legend) 등록 로직 ---
            if k in ("lane_left", "lane_center", "lane_right"):
                # lane_center의 경우 '검은색'인 객체에만 라벨을 붙여서 범례가 검정색이 되게 함
                if k == "lane_center":
                    if not label_set and col == "black":
                        ax.plot(pts[:, 0], pts[:, 1], lw=2.5, color=col, label=k)
                        label_set = True
                    else:
                        ax.plot(pts[:, 0], pts[:, 1], lw=2.5, color=col)
                else:
                    # lane_left, lane_right는 첫 번째 선에 무조건 라벨 등록
                    if not label_set:
                        ax.plot(pts[:, 0], pts[:, 1], lw=2.5, color=col, label=k)
                        label_set = True
                    else:
                        ax.plot(pts[:, 0], pts[:, 1], lw=2.5, color=col)

            elif k == "stop_line":
                ax.plot(pts[:, 0], pts[:, 1], lw=3.0, color="red", ls="--", label=k if not label_set else "")
                label_set = True
            elif k == "car_box":
                poly = _order_polygon_ccw(pts[:4])
                ax.plot(np.r_[poly[:,0], poly[0,0]], np.r_[poly[:,1], poly[0,1]], lw=2.5, color="lime", label=k if not label_set else "")
                label_set = True

    # Reference Path
    ref_uv = np.array([_meter_to_bev_pixel(x, y, xlim, ylim, bev_size) for x, y in ref_path_xy])
    ax.plot(ref_uv[:, 0], ref_uv[:, 1], lw=2, label="reference path")

    # Dynamic Artists
    traj_scatter = ax.scatter([], [], s=20, facecolors='none', edgecolors='red', label="trajectory")
    ego_pt = ax.scatter([], [], s=90, marker="+")
    heading_line, = ax.plot([], [], lw=2)
    ax.legend(loc="upper right")

    # 3. Camera Panel 설정
    ax_cam = fig.add_axes([0.62, 0.14, 0.36, 0.82])
    K = CameraIntrinsics(fx=900.0, fy=900.0, cx=640.0, cy=360.0, width=1280, height=720)
    ax_cam.set_xlim(0, K.width); ax_cam.set_ylim(K.height, 0)
    ax_cam.grid(True)
    
    CAM_COLORS = {"lane_left": "gold", "lane_right": "deepskyblue", "car_box": "lime", "stop_line": "red"}
    cam_artists = []

    # 4. Animation & Interactivity
    idx, playing = {"i": 0}, {"on": False}

    def draw_frame(i):
        i = int(np.clip(i, 0, len(states) - 1))
        idx["i"] = i
        s = states[i]

        # BEV Update
        traj_uv = np.array([_meter_to_bev_pixel(x, y, xlim, ylim, bev_size) for x, y in traj_xy_list[:i+1]])
        traj_scatter.set_offsets(traj_uv)
        
        eu, ev = _meter_to_bev_pixel(s.x, s.y, xlim, ylim, bev_size)
        ego_pt.set_offsets([[eu, ev]])

        hx, hy = s.x + 5.0 * np.cos(s.yaw), s.y + 5.0 * np.sin(s.yaw)
        hu, hv = _meter_to_bev_pixel(hx, hy, xlim, ylim, bev_size)
        heading_line.set_data([eu, hu], [ev, hv])

        # Camera Update
        for a in cam_artists: a.remove()
        cam_artists.clear()
        T_cam_world = make_T_cam_world() @ make_T_vehicle_world(s)

        for name, pts3 in world_pts.items():
            if pts3 is None: continue
            items = pts3 if isinstance(pts3, list) else [pts3]
            label_added = False
            
            for poly in items:
                try:
                    poly_array = np.asarray(poly)
                    
                    # --- [수정] Camera View 색상 결정 로직 ---
                    if name == "lane_center":
                        y_mean = poly_array[:, 1].mean()
                        if abs(y_mean) < 0.1: col = "black"
                        else: col = "gold" if y_mean > 0 else "deepskyblue"
                    elif name == "lane_left":
                        col = "gold"
                    elif name == "lane_right":
                        col = "deepskyblue"
                    elif name == "stop_line":
                        col = "red"
                    elif name == "car_box": # [추가] car_box를 초록색(lime)으로 명시
                        col = "lime"
                    else:
                        col = "gray"

                    # 투영 (Projection)
                    uv, _ = project_pinhole(poly_array, T_cam_world, K)
                    if uv.shape[0] == 0: continue

                    # --- 범례(Legend) 등록 ---
                    if name == "lane_center":
                        # lane_center는 검은색일 때만 범례 등록
                        if not label_added and col == "black":
                            sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col, label=name)
                            label_added = True
                        else:
                            sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col)
                    else:
                        # car_box 포함 나머지 객체 범례 등록
                        if not label_added:
                            sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col, label=name)
                            label_added = True
                        else:
                            sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col)
                    
                    cam_artists.append(sc)
                except Exception:
                    continue
        
        if cam_artists: ax_cam.legend(loc="upper right")
        fig.canvas.draw_idle()

    # Event Handlers
    def on_prev(e): playing["on"] = False; draw_frame(idx["i"] - 1)
    def on_next(e): playing["on"] = False; draw_frame(idx["i"] + 1)
    def on_play(e): playing["on"] = not playing["on"]
    def _tick():
        if playing["on"] and idx["i"] < len(states)-1: draw_frame(idx["i"] + 1)
        elif idx["i"] >= len(states)-1: playing["on"] = False

    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(_tick); timer.start()

    # UI Buttons
    btns = [fig.add_axes([0.20, 0.03, 0.10, 0.07]), fig.add_axes([0.32, 0.03, 0.10, 0.07]), fig.add_axes([0.44, 0.03, 0.12, 0.07])]
    b_prev, b_next, b_play = Button(btns[0], "Prev"), Button(btns[1], "Next"), Button(btns[2], "Play/Pause")
    b_prev.on_clicked(on_prev); b_next.on_clicked(on_next); b_play.on_clicked(on_play)

    draw_frame(0)
    plt.show()