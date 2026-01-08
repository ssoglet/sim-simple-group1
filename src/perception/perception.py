from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from src.dynamics.vehicle import VehicleState

@dataclass
class ObstacleDetection:
    """장애물 감지 결과"""
    position: np.ndarray  # [x, y] in world frame
    size: Tuple[float, float]  # (length, width) in meters
    distance: float  # 장애물까지의 거리 (m)
    relative_velocity: float  # 상대 속도 (m/s, 양수면 접근 중)
    ttc: float  # Time To Collision (초, 음수면 접근하지 않음)
    is_in_path: bool  # 경로상에 있는지 여부
    collision_risk: str  # 'none', 'low', 'medium', 'high', 'critical'

class PerceptionModule:
    """
    Perception 모듈: 장애물 감지 및 분석
    새로운 맵 구조(List 형태 차선) 지원
    """
    
    def __init__(
        self,
        detection_range: float = 50.0,  # 감지 범위 (m)
        safety_margin: float = 2.0,  # 안전 여유 (m)
        vehicle_length: float = 4.5,  # 자차 길이 (m)
        vehicle_width: float = 2.0,  # 자차 폭 (m)
    ):
        self.detection_range = detection_range
        self.safety_margin = safety_margin
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
    
    def detect_obstacles(
        self,
        ego_state: VehicleState,
        world_objects: dict,
        ref_path: np.ndarray = None
    ) -> List[ObstacleDetection]:
        """
        장애물 감지 및 분석
        
        Args:
            ego_state: 자차 상태
            world_objects: 월드 객체 딕셔너리
            ref_path: 참조 경로 (N, 2) 배열
        
        Returns:
            감지된 장애물 리스트
        """
        detections = []
        
        # 월드 객체에서 장애물 추출
        if "car_box" in world_objects:
            obstacle_box = world_objects["car_box"]
            if obstacle_box is not None and len(obstacle_box) > 0:
                detection = self._analyze_box_obstacle(
                    ego_state, obstacle_box, ref_path
                )
                if detection is not None:
                    detections.append(detection)
        
        return detections
    
    def _analyze_box_obstacle(
        self,
        ego_state: VehicleState,
        box_points: np.ndarray,  # (8, 3) - 박스의 8개 꼭짓점
        ref_path: np.ndarray = None
    ) -> ObstacleDetection | None:
        """
        박스 형태 장애물 분석
        """
        # 박스의 중심점과 크기 계산
        box_center_3d = np.mean(box_points, axis=0)
        box_center = box_center_3d[:2]  # [x, y]
        
        # 박스 크기 계산
        xs = box_points[:, 0]
        ys = box_points[:, 1]
        length = np.max(xs) - np.min(xs)
        width = np.max(ys) - np.min(ys)
        
        # 장애물 크기 저장 (추가 검사에서 사용)
        obstacle_width = width
        
        # 자차 위치
        ego_pos = np.array([ego_state.x, ego_state.y])
        
        # 거리 계산
        distance_vec = box_center - ego_pos
        distance = np.linalg.norm(distance_vec)
        
        # 감지 범위 확인
        if distance > self.detection_range:
            return None
        
        # 상대 속도 계산 (정적 장애물 가정)
        relative_velocity = -ego_state.v  # 음수면 접근 중
        
        # TTC 계산 (Time To Collision)
        # 경로 방향으로의 거리만 고려
        ego_heading_vec = np.array([np.cos(ego_state.yaw), np.sin(ego_state.yaw)])
        forward_distance = np.dot(distance_vec, ego_heading_vec)
        
        if forward_distance > 0 and ego_state.v > 0.1:
            ttc = forward_distance / ego_state.v
        else:
            ttc = -1.0  # 접근하지 않음
        
        # 경로상에 있는지 확인
        is_in_path = False
        if ref_path is not None and len(ref_path) > 0:
            is_in_path = self._check_if_obstacle_in_path(
                ego_pos, box_center, box_points, ref_path, ego_state
            )
        
        # 추가 검사: 장애물이 차량 앞쪽에 있고 경로와 겹치는지 확인
        if not is_in_path and forward_distance > 0 and forward_distance < 50.0:
            # 경로의 y 범위 확인
            if len(ref_path) > 0:
                # 경로 앞부분의 y 좌표 범위
                path_y_values = []
                for i, path_pt in enumerate(ref_path):
                    if i < 50:  # 앞 50개 점만 확인
                        path_dist = np.linalg.norm(path_pt[:2] - ego_pos)
                        if 0 < path_dist < forward_distance + 10.0:
                            path_y_values.append(path_pt[1])
                
                if path_y_values:
                    path_y_min = min(path_y_values)
                    path_y_max = max(path_y_values)
                    obs_y_range = (box_center[1] - obstacle_width/2 - self.vehicle_width/2,
                                  box_center[1] + obstacle_width/2 + self.vehicle_width/2)
                    
                    # y 범위가 겹치면 경로상에 있음
                    if not (path_y_max < obs_y_range[0] or path_y_min > obs_y_range[1]):
                        is_in_path = True
        
        # 충돌 위험도 평가
        collision_risk = self._evaluate_collision_risk(
            distance, forward_distance, ttc, is_in_path, relative_velocity
        )
        
        return ObstacleDetection(
            position=box_center,
            size=(length, width),
            distance=distance,
            relative_velocity=relative_velocity,
            ttc=ttc,
            is_in_path=is_in_path,
            collision_risk=collision_risk
        )
    
    def _check_if_obstacle_in_path(
        self,
        ego_pos: np.ndarray,
        obstacle_center: np.ndarray,
        obstacle_points: np.ndarray,
        ref_path: np.ndarray,
        ego_state: VehicleState
    ) -> bool:
        """
        장애물이 경로상에 있는지 확인 (개선된 버전)
        """
        # 자차 앞으로의 경로 부분만 확인 (거리 기반)
        path_ahead_length = 50.0  # 감지 범위 확대
        
        # 장애물의 y 범위 계산
        obs_y_min = obstacle_center[1] - 1.0  # 장애물 폭의 절반 가정
        obs_y_max = obstacle_center[1] + 1.0
        
        # 경로상의 각 점에서 장애물까지의 거리 확인
        for i in range(len(ref_path) - 1):
            path_point = ref_path[i]
            path_next = ref_path[i + 1]
            
            # 자차에서 경로 점까지의 거리
            dist_to_path_point = np.linalg.norm(path_point[:2] - ego_pos)
            
            # 너무 멀거나 뒤에 있으면 스킵
            if dist_to_path_point > path_ahead_length or dist_to_path_point < -5.0:
                continue
            
            # 경로 세그먼트에서 장애물 박스까지의 최단 거리
            min_dist = self._distance_to_box(
                path_point[:2], path_next[:2], obstacle_points
            )
            
            # 경로 폭 고려 (차량 폭 + 안전 여유)
            path_width = self.vehicle_width + self.safety_margin
            
            # 경로가 장애물과 충돌하는지 확인
            if min_dist < path_width:
                # 추가로 경로의 y 좌표 범위와 장애물의 y 범위가 겹치는지 확인
                path_y_min = min(path_point[1], path_next[1])
                path_y_max = max(path_point[1], path_next[1])
                
                # y 범위가 겹치면 경로상에 장애물이 있음
                if not (path_y_max < obs_y_min or path_y_min > obs_y_max):
                    return True
        
        return False
    
    def _distance_to_box(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        box_points: np.ndarray
    ) -> float:
        """
        선분에서 박스까지의 최단 거리
        """
        min_dist = float('inf')
        
        # 박스의 각 변 확인
        box_2d = box_points[:, :2]
        
        # 박스의 4개 모서리 (하단 4개만 사용)
        corners = [
            box_2d[0], box_2d[1], box_2d[2], box_2d[3]
        ]
        
        # 선분에서 각 모서리까지의 거리
        for corner in corners:
            dist = self._point_to_segment_distance(corner, p1, p2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(
        self,
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray
    ) -> float:
        """
        점에서 선분까지의 최단 거리
        """
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-6:
            return np.linalg.norm(point_vec)
        
        t = np.clip(np.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0)
        proj = seg_start + t * seg_vec
        return np.linalg.norm(point - proj)
    
    def _evaluate_collision_risk(
        self,
        distance: float,
        forward_distance: float,
        ttc: float,
        is_in_path: bool,
        relative_velocity: float
    ) -> str:
        """
        충돌 위험도 평가
        """
        if not is_in_path or forward_distance < 0:
            return 'none'
        
        if forward_distance < 3.0 and relative_velocity < -1.0:
            return 'critical'
        elif forward_distance < 5.0 and (ttc > 0 and ttc < 2.0):
            return 'high'
        elif forward_distance < 10.0 and (ttc > 0 and ttc < 5.0):
            return 'medium'
        elif forward_distance < 20.0 and is_in_path:
            return 'low'
        else:
            return 'none'
