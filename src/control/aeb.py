from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.dynamics.vehicle import VehicleState, VehicleParams
from src.perception.perception import ObstacleDetection

@dataclass(frozen=True)
class AEBParams:
    """AEB (Autonomous Emergency Braking) 파라미터"""
    ttc_threshold_warning: float = 5.0  # 경고 TTC (초)
    ttc_threshold_braking: float = 3.0  # 제동 시작 TTC (초)
    ttc_threshold_emergency: float = 1.5  # 긴급 제동 TTC (초)
    min_distance_warning: float = 15.0  # 경고 최소 거리 (m)
    min_distance_braking: float = 10.0  # 제동 최소 거리 (m)
    min_distance_emergency: float = 5.0  # 긴급 제동 최소 거리 (m)
    decel_warning: float = -1.5  # 경고 감속 (m/s^2)
    decel_braking: float = -3.0  # 일반 제동 (m/s^2)
    decel_emergency: float = -6.0  # 긴급 제동 (m/s^2)
    reaction_time: float = 0.5  # 반응 시간 (초)

class AEBModule:
    """
    Rule-based AEB (Autonomous Emergency Braking) 모듈
    TTC(Time To Collision) 및 거리 기반 자동 긴급 제동
    """
    
    def __init__(self, params: AEBParams = None):
        self.params = params or AEBParams()
    
    def compute_safe_acceleration(
        self,
        ego_state: VehicleState,
        obstacles: list[ObstacleDetection],
        current_accel: float
    ) -> tuple[float, dict]:
        """
        안전한 가속도 계산
        
        Args:
            ego_state: 자차 상태
            obstacles: 감지된 장애물 리스트
            current_accel: 현재 제어기에서 계산된 가속도
        
        Returns:
            (안전 가속도, 디버그 정보)
        """
        if not obstacles:
            return current_accel, {"aeb_active": False, "reason": "no_obstacles"}
        
        # 가장 위험한 장애물 찾기
        critical_obstacle = self._find_critical_obstacle(obstacles)
        
        if critical_obstacle is None:
            return current_accel, {"aeb_active": False, "reason": "no_critical"}
        
        # AEB 로직 적용
        safe_accel, reason = self._apply_aeb_logic(
            ego_state, critical_obstacle, current_accel
        )
        
        debug_info = {
            "aeb_active": safe_accel < current_accel,
            "obstacle_distance": critical_obstacle.distance,
            "obstacle_ttc": critical_obstacle.ttc,
            "obstacle_risk": critical_obstacle.collision_risk,
            "original_accel": current_accel,
            "safe_accel": safe_accel,
            "reason": reason
        }
        
        return safe_accel, debug_info
    
    def _find_critical_obstacle(
        self,
        obstacles: list[ObstacleDetection]
    ) -> ObstacleDetection | None:
        """
        가장 위험한 장애물 찾기
        """
        critical_obstacles = [
            obs for obs in obstacles
            if obs.is_in_path and obs.collision_risk in ['high', 'critical']
        ]
        
        if not critical_obstacles:
            return None
        
        # TTC가 가장 작은(가장 위험한) 장애물 선택
        valid_obstacles = [obs for obs in critical_obstacles if obs.ttc > 0]
        
        if not valid_obstacles:
            # TTC가 없으면 거리가 가장 가까운 것 선택
            return min(critical_obstacles, key=lambda obs: obs.distance)
        
        return min(valid_obstacles, key=lambda obs: obs.ttc)
    
    def _apply_aeb_logic(
        self,
        ego_state: VehicleState,
        obstacle: ObstacleDetection,
        current_accel: float
    ) -> tuple[float, str]:
        """
        AEB 로직 적용
        """
        # 현재 속도
        v = ego_state.v
        
        # 거리 및 TTC 기반 제동 결정
        distance = obstacle.distance
        ttc = obstacle.ttc if obstacle.ttc > 0 else float('inf')
        
        # 1. 긴급 제동 조건
        if (distance < self.params.min_distance_emergency or 
            (ttc > 0 and ttc < self.params.ttc_threshold_emergency)):
            safe_accel = self.params.decel_emergency
            reason = "emergency_braking"
        
        # 2. 일반 제동 조건
        elif (distance < self.params.min_distance_braking or 
              (ttc > 0 and ttc < self.params.ttc_threshold_braking)):
            safe_accel = self.params.decel_braking
            reason = "normal_braking"
        
        # 3. 경고 감속 조건
        elif (distance < self.params.min_distance_warning or 
              (ttc > 0 and ttc < self.params.ttc_threshold_warning)):
            safe_accel = max(self.params.decel_warning, current_accel)
            reason = "warning_deceleration"
        
        # 4. 안전한 경우
        else:
            safe_accel = current_accel
            reason = "safe"
        
        # 현재 가속도보다 더 위험한 방향(더 큰 감속)으로만 변경
        safe_accel = min(safe_accel, current_accel)
        
        return safe_accel, reason
