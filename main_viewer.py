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
    bev_dict,
    bev_size,
    xlim,
    ylim,
    ref_path_xy,
    states,          # list of VehicleState
    traj_xy_list,    # list of (x,y) tuples, same length as states
    world_pts,
    title="Day2 Viewer",
):
    
    # --- helpers ---
    def _order_polygon_ccw(pts_uv: np.ndarray) -> np.ndarray:
        """Order 2D points counter-clockwise around centroid."""
        c = pts_uv.mean(axis=0)
        ang = np.arctan2(pts_uv[:,1] - c[1], pts_uv[:,0] - c[0])
        return pts_uv[np.argsort(ang)]

    LANE_COLORS = {
        "lane_left": "gold",
        "lane_center": "black",
        "lane_right": "deepskyblue",
    }    
    
    W, H = bev_size

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


    # --- draw static map (lanes as solid lines, car_box as rectangle) ---
    for k, uv in bev_dict.items():
        if uv is None:
            continue
            
        # OSM 맵의 경우 리스트 형태일 수 있음
        if isinstance(uv, list):
            # 리스트의 각 polyline을 그리기
            if k in ("lane_left", "lane_center", "lane_right"):
                col = LANE_COLORS.get(k, "gray")
                label_set = False
                for polyline_uv in uv:
                    if polyline_uv is None or len(polyline_uv) == 0:
                        continue
                    try:
                        polyline_array = np.asarray(polyline_uv)
                        if polyline_array.ndim == 2 and len(polyline_array) > 0:
                            if not label_set:
                                ax.plot(polyline_array[:, 0], polyline_array[:, 1], 
                                       linewidth=2.5, label=k, color=col)
                                label_set = True
                            else:
                                ax.plot(polyline_array[:, 0], polyline_array[:, 1], 
                                       linewidth=2.5, color=col)
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
            col = LANE_COLORS.get(k, "gray")
            uv_array = np.asarray(uv)
            if uv_array.ndim == 2 and len(uv_array) > 0:
                ax.plot(uv_array[:, 0], uv_array[:, 1], linewidth=2.5, label=k, color=col)
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
        else:
            # 기타 요소들
            uv_array = np.asarray(uv)
            if uv_array.ndim == 2 and len(uv_array) > 0:
                ax.scatter(uv_array[:, 0], uv_array[:, 1], s=3, label=k)

    
    # # draw static map points (once)
    # for k, uv in bev_dict.items():
    #     ax.scatter(uv[:, 0], uv[:, 1], s=3, label=k)

    # draw static reference path (once)
    ref_uv = []
    for (x, y) in ref_path_xy:
        u, v = _meter_to_bev_pixel(x, y, xlim, ylim, bev_size)
        ref_uv.append((u, v))
    ref_uv = np.array(ref_uv, dtype=float)
    ax.plot(ref_uv[:, 0], ref_uv[:, 1], linewidth=2, label="reference path")

    # dynamic artists (will update)

    traj_scatter = ax.scatter([], [], s=20, facecolors='none', edgecolors='red', label="trajectory")
    ego_pt = ax.scatter([], [], s=90, marker="+")
    heading_line, = ax.plot([], [], linewidth=2)
    ax.legend(loc="upper right")

    # cam panel
    ax_cam = fig.add_axes([0.62, 0.14, 0.36, 0.82])

    K = CameraIntrinsics(
        fx=900.0, fy=900.0,
        cx=640.0, cy=360.0,
        width=1280, height=720
    )

    ax_cam.set_title("Pinhole Camera Sensor Viewer")
    ax_cam.set_xlim(0, K.width)
    ax_cam.set_ylim(K.height, 0)
    ax_cam.set_xlabel("u [px]")
    ax_cam.set_ylabel("v [px]")
    ax_cam.grid(True)

    CAM_COLORS = {
        "lane_left": "gold",
        "lane_center": "black",
        "lane_right": "deepskyblue",
        "car_box": "lime",
    }

    cam_artists = []  # frame마다 지우고 다시 그림


    # state for viewer
    idx = {"i": 0}
    playing = {"on": False}
    timer = fig.canvas.new_timer(interval=50)  # ms

    def draw_frame(i):
        i = int(np.clip(i, 0, len(states) - 1))
        idx["i"] = i

        # trajectory up to i
        traj_xy = np.array(traj_xy_list[: i + 1], dtype=float)
        traj_uv = np.array([_meter_to_bev_pixel(x, y, xlim, ylim, bev_size) for x, y in traj_xy])
        traj_scatter.set_offsets(traj_uv)
        
        # ego pose
        s = states[i]
        eu, ev = _meter_to_bev_pixel(s.x, s.y, xlim, ylim, bev_size)
        ego_pt.set_offsets(np.array([[eu, ev]]))

        # heading (yaw)
        Lh = 5.0
        hx = s.x + Lh * np.cos(s.yaw)
        hy = s.y + Lh * np.sin(s.yaw)
        hu, hv = _meter_to_bev_pixel(hx, hy, xlim, ylim, bev_size)
        heading_line.set_data([eu, hu], [ev, hv])

        for a in cam_artists:
            try:
                a.remove()
            except Exception:
                pass
        cam_artists.clear()

        T_cam_world = make_T_cam_world() @ make_T_vehicle_world(s)

        for name, pts3 in world_pts.items():
            if pts3 is None:
                continue
                
            # OSM 맵의 경우 리스트 형태일 수 있음
            if isinstance(pts3, list):
                # 리스트의 각 polyline을 개별적으로 처리
                label_set = False
                for polyline in pts3:
                    if polyline is None or len(polyline) == 0:
                        continue
                    try:
                        polyline_array = np.asarray(polyline)
                        if polyline_array.ndim == 2 and len(polyline_array) > 0:
                            uv, _ = project_pinhole(polyline_array, T_cam_world, K)
                            if uv.shape[0] == 0:
                                continue
                            col = CAM_COLORS.get(name, "gray")
                            if not label_set:
                                sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col, label=name)
                                label_set = True
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
                        col = CAM_COLORS.get(name, "gray")
                        sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col, label=name)
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

