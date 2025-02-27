import os
import cv2
import numpy as np
from ultralytics import YOLO


# Функция для установки необходимых библиотек
def install_dependencies():
    print("Проверка и установка зависимостей...")
    try:
        __import__('ultralytics')
        __import__('opencv-python')
        __import__('numpy')
        print("Все зависимости уже установлены.")
    except ImportError:
        print("Установка необходимых библиотек...")
        os.system('pip install ultralytics opencv-python numpy')
        print("Зависимости успешно установлены.")


# Функция для проверки, пересекается ли bounding box с зоной
def is_box_in_zone(box, area_points_list):
    print("Проверка пересечения bounding box с зонами...")
    x1, y1, x2, y2 = box
    # Преобразуем bounding box в контур
    box_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)

    for area_points in area_points_list:
        # Проверяем пересечение bounding box с зоной
        intersection_area = cv2.intersectConvexConvex(box_points, area_points)[0]
        if intersection_area > 0:
            print("Bounding box пересекается с зоной.")
            return True
    print("Bounding box не пересекается с зонами.")
    return False


# Основная функция обработки
def main():
    print("Запуск основной функции...")
    # Установка зависимостей
    install_dependencies()

    # Загрузка модели YOLOv8
    print("Загрузка модели YOLOv8...")
    model = YOLO('yolov8n.pt')  # yolov8n.pt — предобученная модель YOLOv8
    print("Модель YOLOv8 успешно загружена.")

    # Выбор источника видео
    source_type = input("Выберите тип подключения камеры (usb/rtsp) или введите путь к файлу (видео/фото): ").lower()
    print(f"Выбранный тип подключения: {source_type}")

    if source_type == 'usb':
        # Подключение к USB-камере
        print("Использование USB-камеры...")
        cap = cv2.VideoCapture(0)  # Использование веб-камеры (индекс 0)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть USB-камеру!")
            return
        print("USB-камера успешно открыта.")
    elif source_type == 'rtsp':
        # Подключение к камере по RTSP
        rtsp_url = input("Введите RTSP-URL камеры: ")
        print(f"Подключение к камере по RTSP: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)  # Использование RTSP-камеры
        if not cap.isOpened():
            print("Ошибка: не удалось подключиться к RTSP-камере!")
            return
        print("RTSP-камера успешно подключена.")
    else:
        # Использование видеофайла
        if not os.path.exists(source_type):
            print("Файл не найден!")
            return
        print(f"Использование видеофайла: {source_type}")
        cap = cv2.VideoCapture(source_type)  # Использование видеофайла
        print("Видеофайл успешно открыт.")

    # Создание окна с возможностью изменения размера
    print("Создание окна 'Detection'...")
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    # Установка окна в полноэкранный режим
    cv2.setWindowProperty('Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Окно 'Detection' создано и настроено.")

    # Определение типа файла
    is_video = source_type not in ['usb', 'rtsp'] and source_type.split('.')[-1] in ['mp4', 'avi', 'mov']
    print(f"Тип файла: {'видео' if is_video else 'фото'}")

    # Загрузка первого кадра для выбора области
    print("Загрузка первого кадра...")
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения файла или камеры!")
        return
    print("Первый кадр успешно загружен.")

    # Выбор количества зон
    num_zones = int(input("Введите количество зон для выбора: "))
    print(f"Количество зон: {num_zones}")
    area_points_list = []

    for i in range(num_zones):
        print(f"Выберите область {i + 1} (прямоугольник)")
        area = cv2.selectROI(f"Выбор зоны {i + 1}", frame)
        cv2.destroyWindow(f"Выбор зоны {i + 1}")
        print(f"Область {i + 1} выбрана: {area}")

        # Преобразование области в контур
        x, y, w, h = map(int, area)
        area_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
        area_points_list.append(area_points)
        print(f"Контур области {i + 1} создан: {area_points}")

    while cap.isOpened():
        print("Чтение следующего кадра...")
        ret, frame = cap.read()
        if not ret:
            print("Достигнут конец видео или ошибка чтения.")
            break
        print("Кадр успешно прочитан.")

        # Обработка кадра с помощью YOLOv8
        print("Обработка кадра с помощью YOLOv8...")
        results = model(frame, conf=0.5, classes=[0])  # conf — порог уверенности
        print("Обработка завершена.")

        # Отрисовка результатов
        print("Отрисовка результатов...")
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box
            classes = result.boxes.cls.cpu().numpy()  # Классы объектов
            confidences = result.boxes.conf.cpu().numpy()  # Уверенность

            for box, cls, conf in zip(boxes, classes, confidences):
                if model.names[int(cls)] == 'person':  # Проверяем, что это человек
                    x1, y1, x2, y2 = map(int, box)
                    print(f"Обнаружен человек: {x1}, {y1}, {x2}, {y2}")

                    # Проверка, пересекается ли bounding box с любой из зон
                    if is_box_in_zone((x1, y1, x2, y2), area_points_list):
                        color = (0, 0, 255)  # Красный
                        print("Человек в зоне (красный прямоугольник).")
                    else:
                        color = (0, 255, 0)  # Зеленый
                        print("Человек вне зоны (зеленый прямоугольник).")

                    # Отрисовка прямоугольника
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    print("Прямоугольник нарисован.")

        # Рисуем все выбранные области
        print("Отрисовка выбранных зон...")
        for area_points in area_points_list:
            cv2.polylines(frame, [area_points], True, (255, 0, 0), 2)
        print("Зоны отрисованы.")

        # Получаем размеры окна
        print("Получение размеров окна...")
        window_width = cv2.getWindowImageRect('Detection')[2]
        window_height = cv2.getWindowImageRect('Detection')[3]
        print(f"Размеры окна: {window_width}x{window_height}")

        # Масштабируем изображение на весь экран
        print("Масштабирование изображения...")
        resized_frame = cv2.resize(frame, (window_width, window_height))
        print("Изображение масштабировано.")

        # Отображение кадра
        print("Отображение кадра...")
        cv2.imshow('Detection', resized_frame)
        if cv2.waitKey(1) == 27:  # Выход по нажатию Esc
            print("Выход по нажатию Esc.")
            break

    print("Завершение работы...")
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")


if __name__ == "__main__":
    main()