pip install wiringpi

import wiringpi
import time

# Настройка GPIO
RELAY_1_PIN = 17  # Пин для первого реле (замените на нужный)
RELAY_2_PIN = 27  # Пин для второго реле (замените на нужный)

def setup_gpio():
    """
    Настройка GPIO для управления реле.
    """
    wiringpi.wiringPiSetupGpio()  # Используем нумерацию GPIO (BCM)
    wiringpi.pinMode(RELAY_1_PIN, wiringpi.OUTPUT)
    wiringpi.pinMode(RELAY_2_PIN, wiringpi.OUTPUT)
    print("GPIO настроены.")

def control_relay(relay_pin, state):
    """
    Управление реле.
    :param relay_pin: Пин реле.
    :param state: Состояние (True — включить, False — выключить).
    """
    wiringpi.digitalWrite(relay_pin, wiringpi.HIGH if state else wiringpi.LOW)
    print(f"Реле на пине {relay_pin} {'включено' if state else 'выключено'}.")

def main_relay_control(is_person_in_zone):
    """
    Основная функция управления реле.
    :param is_person_in_zone: Флаг, указывающий, что человек в зоне.
    """
    setup_gpio()

    try:
        while True:
            if is_person_in_zone:
                # Включить реле 1 и выключить реле 2
                control_relay(RELAY_1_PIN, True)
                control_relay(RELAY_2_PIN, False)
            else:
                # Включить реле 2 и выключить реле 1
                control_relay(RELAY_1_PIN, False)
                control_relay(RELAY_2_PIN, True)

            # Задержка для предотвращения слишком частого переключения
            time.sleep(1)

    except KeyboardInterrupt:
        # Выключить все реле при завершении программы
        control_relay(RELAY_1_PIN, False)
        control_relay(RELAY_2_PIN, False)
        print("Программа завершена.")

# Пример использования
if __name__ == "__main__":
    # Пример переменной, которая приходит из основного кода
    is_person_in_zone = False  # Замените на реальное значение из вашего кода
    main_relay_control(is_person_in_zone)


# В вашем основном коде
is_person_in_zone = is_box_in_zone((x1, y1, x2, y2), area_points_list)

# Управление реле
control_relay(RELAY_1_PIN, is_person_in_zone)
control_relay(RELAY_2_PIN, not is_person_in_zone)

#from relay_control import control_relay, RELAY_1_PIN, RELAY_2_PIN



