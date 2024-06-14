import pandas as pd

from PIL import Image
import os
from const import IMG_WIDTH, IMG_HEIGHT



def normalize_coordinates(df, img_width, img_height):
    df['normalized_xcenter'] = df['xcenter'] / img_width
    df['normalized_ycenter'] = df['ycenter'] / img_height
    df['normalized_xwidth'] = df['xwidth'] / img_width
    df['normalized_ywidth'] = df['ywidth'] / img_height
    df['normalized_xheight'] = df['xheight'] / img_width
    df['normalized_yheight'] = df['yheight'] / img_height
    df['normalized_xlength'] = df['xlength'] / img_width
    df['normalized_ylength'] = df['ylength'] / img_height
    return df


def denormalize_coordinates(df, standard_width, standard_height):
    df['scaled_xcenter'] = df['normalized_xcenter'] * standard_width
    df['scaled_ycenter'] = df['normalized_ycenter'] * standard_height
    df['scaled_xwidth'] = df['normalized_xwidth'] * standard_width
    df['scaled_ywidth'] = df['normalized_ywidth'] * standard_height
    df['scaled_xheight'] = df['normalized_xheight'] * standard_width
    df['scaled_yheight'] = df['normalized_yheight'] * standard_height
    df['scaled_xlength'] = df['normalized_xlength'] * standard_width
    df['scaled_ylength'] = df['normalized_ylength'] * standard_height
    return df


def process_images_and_annotations(csv_path, image_dir, output_image_dir, standard_width, standard_height):
    # Зчитування CSV файлу
    df = pd.read_csv(csv_path)

    # Створення вихідних директорій, якщо вони не існують
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # Обробка кожного зображення
    processed_data = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image'])
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Нормалізація координат
        row = normalize_coordinates(pd.DataFrame([row]), img_width, img_height).iloc[0]

        # Масштабування зображення
        image_resized = image.resize((standard_width, standard_height))
        output_image_path = os.path.join(output_image_dir, row['image'])
        image_resized.save(output_image_path)

        # Денормалізація координат для нового розміру
        row = denormalize_coordinates(pd.DataFrame([row]), standard_width, standard_height).iloc[0]

        # Збереження оброблених даних
        processed_data.append([
            row['image'],
            row['scaled_xcenter'], row['scaled_ycenter'],
            row['scaled_xwidth'], row['scaled_ywidth'],
            row['scaled_xheight'], row['scaled_yheight'],
            row['scaled_xlength'], row['scaled_ylength']
        ])

    # Створення нового DataFrame з обробленими даними
    processed_df = pd.DataFrame(processed_data, columns=[
        'image', 'xcenter', 'ycenter', 'xwidth', 'ywidth', 'xheight', 'yheight', 'xlength', 'ylength'
    ])

    return processed_df

def make_csv():
    csv_path = r'C:\Users\PC\PycharmProjects\3D_model_car_object_detection\json3d\output.csv'
    image_dir = r"C:\Users\PC\OneDrive\Робочий стіл\photo for dataset\cars"
    output_image_dir = r'C:\Users\PC\PycharmProjects\3D_model_car_object_detection\json3d\norm_img'

    processed_df = process_images_and_annotations(csv_path, image_dir, output_image_dir, IMG_WIDTH, IMG_HEIGHT)
    processed_df.to_csv(r'C:\Users\PC\PycharmProjects\3D_model_car_object_detection\json3d\processed_annotations.csv', index=False)

    print("Processing complete. Processed annotations saved to 'processed_annotations.csv'.")

make_csv()