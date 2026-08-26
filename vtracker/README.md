# vtracker — переработанная архитектура VBall-Tracker

Чистый слоистый пакет, реализующий целевую архитектуру из
[`../../ARCHITECTURE_REVIEW.md`](../../ARCHITECTURE_REVIEW.md) (§9). Вся
функциональность детекции/трекинга/проекции/визуализации **перенесена нативно**;
прежние реализации (`pipeline.py`, `detectors/`, `trackers/`, `team_detector/`,
`homography/`, `utils/`) удалены. Никаких адаптеров над легаси — только чистые
типобезопасные сущности, интерфейсы и композируемый пайплайн.

## Запуск

```bash
# 1. Скопировать пример конфига и прописать пути под свою машину
cp configs/match.example.yaml configs/match.yaml

# 2. Запустить пайплайн
python -m vtracker.app --config configs/match.yaml

# Архитектурный smoke-тест (без GPU/видео, на фейках):
python -m vtracker.tests.test_pipeline
```

Зависимости — см. [`../requirements.txt`](../requirements.txt). Тяжёлые пакеты
(`ultralytics`, `deep_sort_realtime`, `supervision`, `scikit-learn`) нужны только
для реального запуска; smoke-тест и доменный слой работают без них.

## Слои (зависимости направлены внутрь)

```
app.py                      ← composition root (единственное место с конкретикой)
  │ внедряет (DI)
  ▼
pipeline/  ── stages/, runner.py, context.py   (зависит ТОЛЬКО от domain/interfaces)
  │
domain/    ── entities.py (dataclass), interfaces.py (Protocol)
  ▲
infrastructure/  ── detectors/, trackers/, video/, projection/, exporters/, display.py
visualization/   ── drawers.py (SRP), renderer.py (composition)
core/            ── config.py, logging.py, types.py
```

## Что исправлено относительно старого кода

| Проблема из аудита | Решение в `vtracker` |
|--------------------|----------------------|
| God Object `BallTrackingPipeline.run()` | `PipelineRunner` + список `Stage` (`pipeline/`) |
| Захардкоженные пути в исходниках | `Config.load(yaml)`, пути из файла/env (`core/config.py`) |
| I/O в конструкторе `AppConfig.__post_init__` | конструктор чист, загрузка — явный `Config.load` |
| Магические кортежи `(x,y,w,h,conf)` | `BallDetection` (`domain/entities.py`) |
| Магические словари `player_info`, `track_history` | `Player`, `Referee`, `TrackState` |
| `Visualizer` (484 строки, 7 ответственностей) | `BallDetectionDrawer` / `PeopleDrawer` / `HudDrawer` + `Renderer` |
| ABC объявлены, но не используются для DI | `Protocol`-интерфейсы внедряются в `app.py` |
| `PeopleTracker` нарушает интерфейс `BaseTracker` | единый контракт `PeopleDetector.process()` |
| Дублирующийся логгер-хендлер | идемпотентный `get_logger` (`core/logging.py`) |
| `except Exception: return []` скрывает сбои | стадии логируют `exception()`, не молчат |
| Нельзя тестировать без GPU/видео | `tests/test_pipeline.py` гоняет всё на фейках |

## Как расширять (теперь без правки оркестратора)

- **Новый детектор:** реализуй `BallDetector.detect()` → добавь в `stages` в `app.py`.
- **Новый трекер:** реализуй `BallTracker.update()`.
- **Новый экспорт:** реализуй `FrameExporter` → ещё один `ExportStage`.
- **Новый источник видео (RTSP/папка):** реализуй `VideoSource`.
- **Новый overlay:** новый drawer с `.draw(ctx)` → в список `Renderer`.

## Перенесено нативно (полный список)

- Детекция мяча (motion mask + контуры + YOLO ROI) → `infrastructure/detectors/yolo_ball.py`
- Трекинг мяча (DeepSORT, типизированный `TrackState`, `dt = 1/fps`) → `infrastructure/trackers/deepsort_ball.py`
- Люди (YOLO + ByteTrack) → `infrastructure/detectors/yolo_people.py`
- Команды (KMeans + кеш цвета по `track_id`) → `domain/services/team_assigner.py`
- Гомография (поле + сетка) → `infrastructure/projection/homography.py`
- Визуализация: трек мяча, эллипсы игроков, **мини-карта поля** (кеш базы),
  HUD → `visualization/drawers.py`
- Геометрия/константы поля → `core/geometry.py`, `domain/field.py`

- Разметка точек поля (court/net) для `field_points_path` → `vtracker/tools/field_marker.py`

## Инструменты

### Разметка точек поля

Перед первым запуском на новом видео нужен JSON с точками поля/сетки
(`field_points_path` в конфиге). Разметить их можно интерактивно:

```bash
python -m vtracker.tools.field_marker \
    --video path/to/match.mp4 --frame 195 \
    --width 1280 --height 720 \
    --out data/field_config.json
```

Управление: `1`/`2`/`3` — категория (court/net/other), клик — добавить точку,
`z` — отменить последнюю, `c` — очистить категорию, `s`/`l` — сохранить/загрузить,
`q` — выход. Путей в исходнике нет — всё через аргументы CLI (в отличие от
старого `field_point/field_marker.py`, где путь к видео и конфигу были
захардкожены).

## Возможные следующие шаги

- Мини-карта сетки (side-view) — по образцу `MinimapDrawer`.
- Интерполяция мяча при пропуске детекций (переиспользовать `vball_annotator`).
- Экспорт детекций в JSON (`FrameExporter` → `JsonExporter`).
