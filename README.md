# VBall-Tracker-YOLO-DeepSORT

#ComputerVision #SportsAnalytics #YOLOv11 #DeepSORT #Volleyball

Пайплайн автоматического трекинга волейбольного матча: детекция и трекинг мяча
(YOLO + DeepSORT), игроков и судей (YOLO + ByteTrack) с распределением по командам
(KMeans по цвету формы), проекция на 2D-миникарту через гомографию и рендер
overlay в выходное видео.

> **Архитектура полностью переработана.** Весь код живёт в пакете
> [`vtracker/`](vtracker/README.md) — слоистая Clean Architecture
> (core / domain / infrastructure / pipeline / visualization) с внедрением
> зависимостей и композируемым пайплайном. Прежние реализации
> (`pipeline.py`, `detectors/`, `trackers/`, `team_detector/`, `homography/`,
> `utils/`) удалены — их функциональность перенесена в `vtracker` нативно.
> Подробности и маппинг «проблема → решение» — в
> [`../ARCHITECTURE_REVIEW.md`](../ARCHITECTURE_REVIEW.md).

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
cp configs/match.example.yaml configs/match.yaml   # прописать пути к видео/моделям
python -m vtracker.app --config configs/match.yaml
```

Все пути берутся из YAML-конфига (относительные разрешаются от файла конфига) —
в исходниках захардкоженных путей больше нет.

## Тест архитектуры (без GPU/видео)

```bash
python -m vtracker.tests.test_pipeline      # PIPELINE SMOKE TEST PASSED
```
