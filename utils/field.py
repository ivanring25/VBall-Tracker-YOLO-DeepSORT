from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VolleyballField:
    """
    Адаптированная конфигурация под ваши точки
    """
    court: List[Tuple[int, int]]  # Границы поля как полигон
    net: List[Tuple[int, int]]    # Координаты сетки (4 точки)
    other: List[Tuple[int, int]] = None  # Дополнительные точки

    def validate(self):
        """Проверка минимальных требований"""
        assert len(self.court) >= 4, "Court must have at least 4 boundary points"
        assert len(self.net) == 4, "Net should be defined by 4 points"

    def get_net_rect(self):
        """Возвращает координаты сетки как прямоугольник"""
        return (self.net[0], self.net[2])  # (top-left, bottom-right)

    @property
    def court_center(self):
        """Центр поля как среднее арифметическое всех точек"""
        xs = [p[0] for p in self.court]
        ys = [p[1] for p in self.court]
        return (sum(xs)//len(xs)), (sum(ys)//len(ys))

    def normalize_position(self, point):
        """Нормализация координат относительно центра поля"""
        cx, cy = self.court_center
        return (point[0]-cx, point[1]-cy)