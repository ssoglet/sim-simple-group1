import matplotlib.pyplot as plt

from src.common.types import CameraIntrinsics
from src.world.world import make_demo_world_points
from src.sensors.camera_pinhole import project_pinhole
from src.sensors.camera_extrinsic import make_T_cam_world
from src.sensors.bev import world_xy_to_bev, clip_bev
from src.sensors.viz_sensor import plot_image, plot_bev

def main():
    world_pts = make_demo_world_points()

    K = CameraIntrinsics(
        fx=900, fy=900,
        cx=640, cy=360,
        width=1280, height=720,
        # distortion = (k0,k1,k2,k3,k4),
        # undistortion = (u0,u1,u2,u3,u4),
    )

    T_cam_world = make_T_cam_world()

    uv_dict = {}
    for name, pts in world_pts.items():
        uv, _ = project_pinhole(pts, T_cam_world, K)
        uv_dict[name] = uv

    bev_size = (600,600)
    bev_dict = {}
    for name, pts in world_pts.items():
        bev_uv = world_xy_to_bev(
            pts,
            xlim=(0,60),
            ylim=(-10,10),
            bev_size=bev_size
        )
        bev_dict[name] = clip_bev(bev_uv, bev_size)

    plot_image(uv_dict, K.width, K.height)
    plot_bev(
        bev_dict,
        bev_size=bev_size,
        xlim=(0, 60),
        ylim=(-10, 10)
    )
    plt.show()

if __name__ == "__main__":
    main()

# Camera model이 pinhole with no distortion 만 있는데,
# -> distortion, undistortion 추가 가능
# -> fisheye
# Camera Extrinsic 을 바꿔도 됨
# -> x,y,z,roll,pitch,yaw

# Map(world) 변경 가능
# -> lane
# -> traffic (car, human) -> (car : 선행차량/아닌 차량), (human : 도로 위에 있는 사람, sidewalk에 잇는 사람..)
# perception model -> tracking tajectory 인지 모듈, ....(다양)
# -> traffic sign ( -> path planning에서 굉장히 중요하게 보는 perception input)
# perception model -> 신호등이랑 정지선 같이 법규적으로 인지가능한 Signal들을 인지하는 모듈 .... <- Rule-based 에서는 점점 성능이 좋아지겠죠.

# reference path 생성 -> planning 할 때 perception 출력을 사용함
# v2x 로 신호등 신호 받아오기도 함(ex. 서울시청 신호등 정보 실시간으로 통신해서 signal 주세요)