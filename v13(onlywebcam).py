import os
import cv2
import numpy as np
from ultralytics import YOLO


# Функция для установки необходимых библиотек
def install_dependencies():
    try:
        __import__('ultralytics')
        __import__('opencv-python')
        __import__('numpy')
    except ImportError:
        print("Установка необходимых библиотек...")
        os.system('pip install ultralytics opencv-python numpy')


# Функция для проверки, пересекается ли bounding box с зоной
def is_box_in_zone(box, area_points):
    x1, y1, x2, y2 = box

    # Преобразуем bounding box в контур в формате (N, 1, 2)
    box_contour = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32)

    # Преобразуем area_points в формат (N, 1, 2)
    area_points = area_points.reshape((-1, 1, 2))

    # Проверяем пересечение с помощью intersectConvexConvex
    intersection_area, _ = cv2.intersectConvexConvex(box_contour, area_points)
    return intersection_area > 0  # Возвращаем True, если площадь пересечения больше 0


# Функция для выбора зоны по точкам
def select_area(event, x, y, flags, param):
    global points, frame_copy, area_points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Добавляем точку
        if len(points) > 1:
            # Рисуем линию между последними двумя точками
            cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Выбор зоны", frame_copy)


# Функция для получения списка доступных камер
def get_available_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
        cap.release()
        index += 1
    return cameras


# Основная функция обработки
def main():
    global points, frame_copy, area_points

    # Установка зависимостей
    install_dependencies()

    # Загрузка модели YOLOv8
    model = YOLO('yolov8n.pt')  # yolov8n.pt — предобученная модель YOLOv8

    # Выбор источника видео
    source = input("Введите путь к видеофайлу или 'камера' для использования веб-камеры: ")

    if source.lower() == 'камера':
        # Получаем список доступных камер
        cameras = get_available_cameras()
        if not cameras:
            print("Нет доступных камер!")
            return

        # Выводим список камер
        print("Доступные камеры:")
        for i, cam in enumerate(cameras):
            print(f"{i}: Камера {cam}")

        # Пользователь выбирает камеру
        cam_index = int(input("Выберите номер камеры: "))
        if cam_index < 0 or cam_index >= len(cameras):
            print("Неверный выбор камеры!")
            return

        # Открываем выбранную камеру
        cap = cv2.VideoCapture(cameras[cam_index])
        if not cap.isOpened():
            print("Ошибка: не удалось открыть камеру!")
            return

        # Создание окна с возможностью изменения размера
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        # Установка окна в полноэкранный режим
        cv2.setWindowProperty('Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        if not os.path.exists(source):
            print("Файл не найден!")
            return
        cap = cv2.VideoCapture(source)  # Использование видеофайла

    # Загрузка первого кадра для выбора области
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения файла или камеры!")
        return

    # Копия кадра для выбора зоны
    frame_copy = frame.copy()
    points = []  # Список для хранения точек
    area_points = None  # Контур зоны

    # Окно для выбора зоны
    cv2.namedWindow("Выбор зоны")
    cv2.setMouseCallback("Выбор зоны", select_area)

    while True:
        cv2.imshow("Выбор зоны", frame_copy)
        key = cv2.waitKey(1) & 0xFF

        # Завершение выбора зоны по нажатию Enter
        if key == 13:  # Код клавиши Enter
            if len(points) >= 3:  # Минимум 3 точки
                area_points = np.array([points], dtype=np.int32)  # Формат (1, N, 2)
                break
            else:
                print("Выберите минимум 3 точки!")
        # Выход по нажатию Esc
        elif key == 27:
            print("Выбор зоны отменен.")
            return

    cv2.destroyWindow("Выбор зоны")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра с помощью YOLOv8
        results = model(frame, conf=0.5, classes=[0])  # conf — порог уверенности

        # Отрисовка результатов
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box
            classes = result.boxes.cls.cpu().numpy()  # Классы объектов
            confidences = result.boxes.conf.cpu().numpy()  # Уверенность

            for box, cls, conf in zip(boxes, classes, confidences):
                if model.names[int(cls)] == 'person':  # Проверяем, что это человек
                    x1, y1, x2, y2 = map(int, box)

                    # Проверка, пересекается ли bounding box с зоной
                    if is_box_in_zone((x1, y1, x2, y2), area_points):
                        color = (0, 0, 255)  # Красный
                        cv2.putText(frame, 'В зоне', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        color = (0, 255, 0)  # Зеленый

                    # Отрисовка прямоугольника
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Рисуем выбранную область
        cv2.polylines(frame, [area_points], True, (255, 0, 0), 2)

        # Отображение кадра
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) == 27:  # Выход по нажатию Esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()