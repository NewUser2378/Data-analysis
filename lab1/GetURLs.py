import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Список URL для обработки
urls = [
    'https://realty.ya.ru/sankt-peterburg/snyat/kvartira/ryadom-metro/',
    'https://realty.ya.ru/sankt-peterburg/snyat/kvartira/ryadom-metro/?page=2',
    'https://realty.ya.ru/sankt-peterburg/snyat/kvartira/ryadom-metro/?page=3',
    'https://realty.ya.ru/sankt-peterburg/snyat/kvartira/ryadom-metro/?page=4',
    'https://realty.ya.ru/sankt-peterburg/snyat/kvartira/ryadom-metro/?page=5',
]
for i in range(2,25):
    link = 'https://realty.ya.ru/sankt-peterburg/snyat/kvartira/ryadom-metro/?page='
    link += str(i)
    urls.append(link)

# Функция для обработки каждой ссылки
def process_url(url):
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)

    try:
        driver.get(url)

        # Ожидание загрузки элементов
        wait = WebDriverWait(driver, 3)

        # Используем множество для хранения уникальных ссылок
        unique_links = set()

        # Прокрутка страницы вниз для загрузки всех ссылок
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Получаем все ссылки на квартиры из блока
            links = wait.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "a.Link.Link_js_inited.Link_size_m.Link_theme_islands")))

            for link in links:
                href = link.get_attribute('href')
                if href and "/offer/" in href:  # Проверяем, содержит ли ссылка '/offer/'
                    unique_links.add(href)  # Добавляем ссылку в множество

            # Прокручиваем страницу вниз
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Ждем загрузки новых элементов

            # Проверяем, прокрутилась ли страница
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:  # Если высота не изменилась, значит, достигли конца
                break
            last_height = new_height

        # Записываем уникальные ссылки в файл
        with open("links.txt", "a", encoding="utf-8") as file:  # Используем режим "a" для добавления
            for link in unique_links:
                file.write(link + "\n")  # Записываем ссылку в файл с новой строки

        print(f"Ссылки успешно записаны для {url}")

    finally:
        driver.quit()


# Обработка каждой ссылки из списка
for url in urls:
    process_url(url)
