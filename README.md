<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.19-green?logo=github)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green?logo=nvidia)

# Детекция сотрудников сети магазинов X5

[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## 📋 Содержание

- [О проекте](#-о-проекте)
- [Постановка задачи](#-постановка-задачи)
- [Архитектура решения](#-архитектура-решения)
- [Технический стек](#-технический-стек)
- [Результаты](#-результаты)
- [Установка и запуск](#-установка-и-запуск)
- [Структура проекта](#-структура-проекта)
- [Ключевые особенности](#-ключевые-особенности)

## 📖 О проекте
Данный репозиторий содержит решение задачи детекции объектов. Основная цель — разработка эффективного пайплайна для локализации людей на изображениях с видеопотока и их классификации на два класса: customer (покупатель) и staff (сотрудник).

> Финальная метрика: точность детекции только класса staff для submission.csv

## 🛠 Технический стек
### Основные фреймворки и библиотеки
<div align="center">
  
  | Компонент | Версия | Назначение |
  |-----------|--------|------------|
  | Python | 3.12.12 | Язык программирования |
  | PyTorch | 2.9.0+cu126 | Deep Learning фреймворк |
  | Ultralytics | 8.4.19 | YOLOv8 архитектура |
  | CUDA | 12.6 | GPU ускорение |
  
</div>

### Ключевые зависимости

```bash
ultralytics==8.4.19      # YOLOv8 модели
sahi                     # Slicing Aided Hyper Inference
ensemble-boxes           # Weighted Box Fusion
pandas                   # Работа с данными
numpy                    # Численные вычисления
tqdm                     # Прогресс-бары
pyyaml                   # YAML конфигурации
albumentations           # Аугментации изображений
```

### Архитектура моделей
<div align="center">

| Модель | Размер | Image Size | Epochs | mAP |
|--------|--------|------------|--------|-----|
| YOLOv8n | Nano | 640 | 25 | ~0.769 |
| YOLOv8m | Medium | 768 | 40 | ~0.792 |
| YOLOv8l | Large | 768 | 40 | ~0.845 |

</div>

### Методы улучшения качества
- **Ensemble**: Weighted Box Fusion (WBF) с параметрами `iou_thr=0.5`, `skip_box_thr=0.001`
- **SAHI**: Slicing Aided Hyper Inference для детекции мелких объектов
- **Multi-scale**: Инференс на разных размерах изображений (640, 768)
- **Checkpoints**: Сохранение и возобновление обучения (best.pt)
