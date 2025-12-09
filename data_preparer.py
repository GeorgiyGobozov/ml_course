# data_preparer.py - модуль подготовки данных
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

class DataPreparer:
    """
    Класс для подготовки пользовательского датасета
    """
    
    def __init__(self, dataset_name='my_dataset', img_size=(64, 64), grayscale=True):
        """
        Инициализация подготовителя данных
        
        Args:
            dataset_name (str): Имя датасета
            img_size (tuple): Размер изображений (ширина, высота)
            grayscale (bool): Преобразовывать в градации серого
        """
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.grayscale = grayscale
        self.base_dir = Path(dataset_name)
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Создаем структуру директорий
        self._create_directories()
    
    def _create_directories(self):
        """Создает структуру директорий для датасета"""
        directories = ['raw', 'processed', 'train', 'val', 'test', 'metadata']
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Структура датасета создана в: {self.base_dir}")
    
    def organize_data_interactive(self):
        """Интерактивная организация данных"""
        print("\n" + "="*60)
        print("ОРГАНИЗАЦИЯ ДАННЫХ")
        print("="*60)
        
        # Запрос количества классов
        while True:
            try:
                num_classes = int(input("Введите количество классов: "))
                if num_classes > 0:
                    break
                else:
                    print("Количество классов должно быть положительным числом.")
            except ValueError:
                print("Пожалуйста, введите целое число.")
        
        # Запрос названий классов
        self.classes = []
        for i in range(num_classes):
            while True:
                class_name = input(f"Введите название класса {i+1}: ").strip()
                if class_name:
                    self.classes.append(class_name)
                    self.class_to_idx[class_name] = i
                    self.idx_to_class[i] = class_name
                    
                    # Создаем директорию для класса
                    class_dir = self.base_dir / 'raw' / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Создана папка: {class_dir}")
                    print(f"Поместите изображения в формате JPG, PNG, BMP в эту папку")
                    input("Нажмите Enter, когда добавите изображения...")
                    break
                else:
                    print("Название класса не может быть пустым.")
        
        print(f"\nКлассы успешно созданы: {', '.join(self.classes)}")
        return self.classes
    
    def process_images(self):
        """Обработка и подготовка изображений"""
        print("\n" + "="*60)
        print("ОБРАБОТКА ИЗОБРАЖЕНИЙ")
        print("="*60)
        
        raw_dir = self.base_dir / 'raw'
        processed_dir = self.base_dir / 'processed'
        
        image_info = []
        total_images = 0
        
        for class_name in self.classes:
            class_raw_dir = raw_dir / class_name
            class_processed_dir = processed_dir / class_name
            class_processed_dir.mkdir(exist_ok=True)
            
            # Получаем все изображения
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(class_raw_dir.glob(ext))
            
            if len(image_files) == 0:
                print(f"⚠️  Внимание: Нет изображений в классе '{class_name}'")
                continue
            
            print(f"Обработка класса '{class_name}' ({len(image_files)} изображений)...")
            
            for img_file in tqdm(image_files, desc=f"Класс {class_name}"):
                try:
                    # Открываем изображение
                    with Image.open(img_file) as img:
                        original_size = img.size
                        
                        # Конвертируем в нужный формат
                        if self.grayscale:
                            img = img.convert('L')  # Градации серого
                        else:
                            img = img.convert('RGB')  # Цветное
                        
                        # Изменяем размер
                        img_resized = img.resize(self.img_size, Image.Resampling.LANCZOS)
                        
                        # Сохраняем
                        output_path = class_processed_dir / f"{img_file.stem}.png"
                        img_resized.save(output_path, 'PNG')
                        
                        image_info.append({
                            'original_path': str(img_file),
                            'processed_path': str(output_path),
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name],
                            'original_size': original_size,
                            'processed_size': self.img_size,
                            'grayscale': self.grayscale
                        })
                        
                        total_images += 1
                        
                except Exception as e:
                    print(f"Ошибка при обработке {img_file}: {e}")
        
        # Сохраняем информацию об изображениях
        if image_info:
            df = pd.DataFrame(image_info)
            metadata_dir = self.base_dir / 'metadata'
            df.to_csv(metadata_dir / 'image_info.csv', index=False)
            
            print(f"\n✅ Обработка завершена!")
            print(f"Всего обработано изображений: {total_images}")
            
            # Создаем визуализацию
            self._create_visualization(df)
        
        return total_images
    
    def _create_visualization(self, df):
        """Создает визуализацию данных"""
        # Распределение по классам
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        class_counts = df['class'].value_counts()
        bars = plt.bar(class_counts.index, class_counts.values)
        plt.title('Распределение изображений по классам')
        plt.xlabel('Класс')
        plt.ylabel('Количество')
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        if self.grayscale:
            plt.hist(np.random.randn(1000), bins=30, alpha=0.7, color='gray')
        else:
            colors = ['red', 'green', 'blue']
            for i in range(3):
                plt.hist(np.random.randn(1000), bins=30, alpha=0.3, color=colors[i])
        plt.title('Пример гистограммы')
        plt.xlabel('Интенсивность')
        plt.ylabel('Частота')
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'metadata' / 'data_distribution.png', dpi=150)
        plt.show()
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Разделение датасета на выборки"""
        print("\n" + "="*60)
        print("РАЗДЕЛЕНИЕ НА ВЫБОРКИ")
        print("="*60)
        
        processed_dir = self.base_dir / 'processed'
        
        split_info = []
        
        for class_name in self.classes:
            class_dir = processed_dir / class_name
            image_files = list(class_dir.glob('*.png'))
            
            if len(image_files) == 0:
                print(f"⚠️  Пропуск класса '{class_name}' - нет изображений")
                continue
            
            # Разделяем файлы
            train_files, temp_files = train_test_split(
                image_files, train_size=train_ratio, random_state=42
            )
            
            val_files, test_files = train_test_split(
                temp_files,
                train_size=val_ratio/(val_ratio + test_ratio),
                random_state=42
            )
            
            # Копируем в соответствующие директории
            for split_type, files in [('train', train_files),
                                     ('val', val_files),
                                     ('test', test_files)]:
                split_class_dir = self.base_dir / split_type / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for file_path in files:
                    shutil.copy2(file_path, split_class_dir / file_path.name)
                    split_info.append({
                        'path': str(split_class_dir / file_path.name),
                        'class': class_name,
                        'split_type': split_type
                    })
            
            print(f"Класс '{class_name}':")
            print(f"  Обучающая: {len(train_files)}")
            print(f"  Валидационная: {len(val_files)}")
            print(f"  Тестовая: {len(test_files)}")
        
        # Сохраняем информацию о разделении
        if split_info:
            split_df = pd.DataFrame(split_info)
            metadata_dir = self.base_dir / 'metadata'
            split_df.to_csv(metadata_dir / 'split_info.csv', index=False)
            
            print(f"\n✅ Разделение завершено!")
            print(f"Всего изображений распределено: {len(split_info)}")
    
    def create_numpy_dataset(self):
        """Создает numpy датасет для обучения"""
        print("\n" + "="*60)
        print("СОЗДАНИЕ ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ")
        print("="*60)
        
        # Загружаем данные из всех выборок
        X_train, y_train = self._load_split_data('train')
        X_val, y_val = self._load_split_data('val')
        X_test, y_test = self._load_split_data('test')
        
        # Проверяем, что данные есть
        if len(X_train) == 0:
            print("❌ Ошибка: Нет данных для обучения!")
            return None
        
        # Сохраняем в npz файл
        output_file = self.base_dir / 'dataset.npz'
        np.savez_compressed(
            output_file,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        
        # Сохраняем метаданные
        metadata = {
            'dataset_name': self.dataset_name,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'image_size': self.img_size,
            'grayscale': self.grayscale,
            'num_train': len(X_train),
            'num_val': len(X_val),
            'num_test': len(X_test),
            'input_shape': X_train[0].shape
        }
        
        metadata_dir = self.base_dir / 'metadata'
        with open(metadata_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Датсет создан!")
        print(f"Файл: {output_file}")
        print(f"Размеры данных:")
        print(f"  Обучающая выборка: {X_train.shape}")
        print(f"  Валидационная выборка: {X_val.shape}")
        print(f"  Тестовая выборка: {X_test.shape}")
        print(f"  Размер одного изображения: {X_train[0].shape}")
        
        return output_file
    
    def _load_split_data(self, split_name):
        """Загружает данные из указанной выборки"""
        split_dir = self.base_dir / split_name
        X = []
        y = []
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.png'):
                    try:
                        with Image.open(img_file) as img:
                            img_array = np.array(img)
                        
                        # Нормализуем значения от 0 до 1
                        img_normalized = img_array.astype(np.float32) / 255.0
                        
                        # Если градации серого и массив 2D, добавляем размерность канала
                        if self.grayscale and len(img_normalized.shape) == 2:
                            img_normalized = np.expand_dims(img_normalized, axis=-1)
                        
                        X.append(img_normalized)
                        y.append(self.class_to_idx[class_name])
                        
                    except Exception as e:
                        print(f"Ошибка при загрузке {img_file}: {e}")
        
        return np.array(X), np.array(y)
    
    def run_full_pipeline(self):
        """Запускает полный конвейер подготовки данных"""
        print("="*70)
        print("ПОЛНЫЙ КОНВЕЙЕР ПОДГОТОВКИ ДАННЫХ")
        print("="*70)
        
        # 1. Организация данных
        self.organize_data_interactive()
        
        # 2. Обработка изображений
        self.process_images()
        
        # 3. Разделение на выборки
        self.split_dataset()
        
        # 4. Создание numpy датасета
        dataset_file = self.create_numpy_dataset()
        
        print("\n" + "="*70)
        print("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
        print("="*70)
        
        return dataset_file


# Функция для быстрого создания датасета
def create_dataset():
    """Основная функция для запуска подготовки данных"""
    print("ПРОГРАММА ПОДГОТОВКИ ДАННЫХ")
    print("="*60)
    
    # Запрос параметров
    print("\nВведите параметры датасета:")
    
    dataset_name = input("Название датасета (по умолчанию: my_dataset): ") or "my_dataset"
    
    while True:
        try:
            width = int(input("Ширина изображений (по умолчанию: 64): ") or "64")
            height = int(input("Высота изображений (по умолчанию: 64): ") or "64")
            if width > 0 and height > 0:
                break
            else:
                print("Размеры должны быть положительными числами.")
        except ValueError:
            print("Пожалуйста, введите целые числа.")
    
    grayscale_input = input("Градации серого? (y/n, по умолчанию: y): ").lower()
    grayscale = grayscale_input != 'n'
    
    # Создаем и запускаем подготовитель
    preparer = DataPreparer(
        dataset_name=dataset_name,
        img_size=(width, height),
        grayscale=grayscale
    )
    
    # Запускаем конвейер
    dataset_file = preparer.run_full_pipeline()
    
    if dataset_file:
        print(f"\n📁 Ваш датасет готов к использованию!")
        print(f"Путь к датасету: {dataset_file}")
        print(f"\nДля обучения моделей выполните:")
        print(f"python model_comparison.py --dataset {dataset_file}")
    
    return dataset_file


if __name__ == "__main__":
    create_dataset()