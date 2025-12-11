# data_preparer.py - модуль подготовки данных
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from pathlib import Path
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import math
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class RobustAugmentationPipeline:
    """
    Надежный конвейер аугментации изображений без использования putdata()
    """
    
    def __init__(self, enable_augmentation=True, augmentation_multiplier=10):
        """
        Инициализация конвейера аугментации
        
        Args:
            enable_augmentation (bool): Включить аугментацию
            augmentation_multiplier (int): Во сколько раз увеличить датасет
        """
        self.enable_augmentation = enable_augmentation
        self.augmentation_multiplier = augmentation_multiplier
    
    def apply_augmentation(self, image):
        """
        Применяет случайную аугментацию к изображению
        Возвращает numpy array
        """
        if not self.enable_augmentation:
            return image
        
        # Конвертируем в PIL Image если нужно
        original_is_numpy = isinstance(image, np.ndarray)
        if original_is_numpy:
            # Сохраняем оригинальную форму
            original_shape = image.shape
            
            # Конвертируем в PIL
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3 and image.shape[2] == 1:
                # Градации серого
                pil_image = Image.fromarray(image[:, :, 0], mode='L')
                is_grayscale = True
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Цветное RGB
                pil_image = Image.fromarray(image, mode='RGB')
                is_grayscale = False
            elif len(image.shape) == 2:
                # 2D градации серого
                pil_image = Image.fromarray(image, mode='L')
                is_grayscale = True
            else:
                # Неподдерживаемый формат
                return image
        else:
            # Уже PIL Image
            pil_image = image
            is_grayscale = (pil_image.mode == 'L')
        
        # Выбираем случайную аугментацию
        aug_type = random.choice([
            'flip_horizontal', 'flip_vertical', 'rotate', 
            'brightness', 'contrast', 'color_jitter',
            'gaussian_blur', 'gaussian_noise', 'zoom',
            'translation', 'shear'
        ])
        
        # Применяем выбранную аугментацию
        try:
            if aug_type == 'flip_horizontal':
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif aug_type == 'flip_vertical':
                pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            elif aug_type == 'rotate':
                angle = random.uniform(-15, 15)
                pil_image = pil_image.rotate(angle, resample=Image.BICUBIC, expand=False)
            elif aug_type == 'brightness':
                factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(factor)
            elif aug_type == 'contrast':
                factor = random.uniform(0.7, 1.3)
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(factor)
            elif aug_type == 'color_jitter' and not is_grayscale:
                # Применяем несколько цветовых коррекций
                pil_image = self._apply_color_jitter(pil_image)
            elif aug_type == 'gaussian_blur':
                radius = random.uniform(0.5, 1.5)
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            elif aug_type == 'gaussian_noise':
                pil_image = self._apply_gaussian_noise(pil_image)
            elif aug_type == 'zoom':
                scale = random.uniform(0.8, 1.2)
                pil_image = self._apply_zoom(pil_image, scale)
            elif aug_type == 'translation':
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                pil_image = self._apply_translation(pil_image, dx, dy)
            elif aug_type == 'shear':
                shear_x = random.uniform(-0.1, 0.1)
                shear_y = random.uniform(-0.1, 0.1)
                pil_image = self._apply_shear(pil_image, shear_x, shear_y)
        except Exception as e:
            print(f"Ошибка при аугментации: {e}")
            # Возвращаем оригинальное изображение в случае ошибки
            if original_is_numpy:
                return image
            else:
                return pil_image
        
        # Конвертируем обратно в numpy если нужно
        if original_is_numpy:
            result_array = np.array(pil_image)
            
            # Восстанавливаем оригинальную форму
            if is_grayscale and len(original_shape) == 3 and original_shape[2] == 1:
                if len(result_array.shape) == 2:
                    result_array = np.expand_dims(result_array, axis=-1)
            
            # Нормализуем обратно к [0, 1]
            if result_array.dtype != np.float32:
                result_array = result_array.astype(np.float32) / 255.0
            
            return result_array
        else:
            return pil_image
    
    def _apply_color_jitter(self, image):
        """Применяет цветовую аугментацию"""
        # Яркость
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # Контраст
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # Насыщенность
        saturation_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        
        return image
    
    def _apply_gaussian_noise(self, image):
        """Добавляет гауссовский шум"""
        if image.mode == 'L':
            array = np.array(image, dtype=np.float32)
            noise = np.random.normal(0, 10, array.shape)
            noisy = np.clip(array + noise, 0, 255)
            return Image.fromarray(noisy.astype(np.uint8))
        else:
            # Обрабатываем каждый канал отдельно
            channels = []
            for channel in image.split():
                array = np.array(channel, dtype=np.float32)
                noise = np.random.normal(0, 10, array.shape)
                noisy = np.clip(array + noise, 0, 255)
                channels.append(Image.fromarray(noisy.astype(np.uint8)))
            return Image.merge(image.mode, channels)
    
    def _apply_zoom(self, image, scale):
        """Применяет масштабирование"""
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Масштабируем
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        if scale > 1:
            # Обрезаем центр
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            return resized.crop((left, top, right, bottom))
        else:
            # Размещаем по центру на черном фоне
            if image.mode == 'L':
                bg_color = 0
            else:
                bg_color = (0, 0, 0)
            
            result = Image.new(image.mode, (width, height), color=bg_color)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            result.paste(resized, (left, top))
            return result
    
    def _apply_translation(self, image, dx, dy):
        """Применяет сдвиг"""
        width, height = image.size
        
        if image.mode == 'L':
            bg_color = 0
        else:
            bg_color = (0, 0, 0)
        
        result = Image.new(image.mode, (width, height), color=bg_color)
        result.paste(image, (dx, dy))
        return result
    
    def _apply_shear(self, image, shear_x, shear_y):
        """Применяет скол"""
        width, height = image.size
        
        # Создаем матрицу преобразования
        matrix = (1, shear_x, 0,
                 shear_y, 1, 0)
        
        return image.transform((width, height), Image.AFFINE, matrix, 
                              resample=Image.BICUBIC)
    
    def augment_dataset(self, images, labels, multiplier=None):
        """
        Аугментирует датасет
        
        Args:
            images: Массив изображений (numpy array)
            labels: Массив меток
            multiplier: Во сколько раз увеличить датасет
        
        Returns:
            Augmented images and labels
        """
        if not self.enable_augmentation:
            return images, labels
        
        if multiplier is None:
            multiplier = self.augmentation_multiplier
        
        augmented_images = []
        augmented_labels = []
        
        print(f"Аугментация данных (увеличение в {multiplier} раз)...")
        
        # Для каждого изображения создаем несколько аугментированных версий
        for i in tqdm(range(len(images)), desc="Аугментация"):
            original_img = images[i]
            label = labels[i]
            
            # Добавляем оригинальное изображение
            augmented_images.append(original_img)
            augmented_labels.append(label)
            
            # Создаем аугментированные версии
            for j in range(multiplier - 1):
                # Применяем несколько аугментаций (1-3)
                num_augs = random.randint(1, 3)
                aug_img = original_img.copy()
                
                for _ in range(num_augs):
                    aug_img = self.apply_augmentation(aug_img)
                
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)


class DataPreparer:
    """
    Класс для подготовки пользовательского датасета с аугментацией
    """
    
    def __init__(self, dataset_name='my_dataset', img_size=(64, 64), grayscale=True,
                 enable_augmentation=True, augmentation_multiplier=10):
        """
        Инициализация подготовителя данных
        
        Args:
            dataset_name (str): Имя датасета
            img_size (tuple): Размер изображений (ширина, высота)
            grayscale (bool): Преобразовывать в градации серого
            enable_augmentation (bool): Включить аугментацию
            augmentation_multiplier (int): Во сколько раз увеличить датасет
        """
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.grayscale = grayscale
        self.enable_augmentation = enable_augmentation
        self.augmentation_multiplier = augmentation_multiplier
        
        self.base_dir = Path(dataset_name)
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Инициализация конвейера аугментации
        self.augmentation_pipeline = RobustAugmentationPipeline(
            enable_augmentation=enable_augmentation,
            augmentation_multiplier=augmentation_multiplier
        )
        
        # Создаем структуру директорий
        self._create_directories()
    
    def _create_directories(self):
        """Создает структуру директорий для датасета"""
        directories = ['raw', 'processed', 'train', 'val', 'test', 
                      'metadata', 'augmented_train', 'augmented_val']
        
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
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
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
                            mode = 'L'
                        else:
                            img = img.convert('RGB')  # Цветное
                            mode = 'RGB'
                        
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
                            'mode': mode
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
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
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
        
        plt.subplot(2, 2, 2)
        sizes = df['original_size'].apply(lambda x: x[0] * x[1])
        plt.hist(sizes, bins=30, alpha=0.7, color='green')
        plt.title('Распределение размеров исходных изображений')
        plt.xlabel('Количество пикселей')
        plt.ylabel('Частота')
        
        plt.subplot(2, 2, 3)
        aspect_ratios = df['original_size'].apply(lambda x: x[0] / x[1])
        plt.hist(aspect_ratios, bins=30, alpha=0.7, color='purple')
        plt.title('Соотношения сторон')
        plt.xlabel('Ширина/Высота')
        plt.ylabel('Частота')
        
        # Показываем пример изображения
        plt.subplot(2, 2, 4)
        try:
            sample_path = df.iloc[0]['processed_path']
            sample_img = Image.open(sample_path)
            
            if self.grayscale:
                plt.imshow(sample_img, cmap='gray')
            else:
                plt.imshow(sample_img)
            
            plt.title('Пример обработанного изображения')
            plt.axis('off')
        except:
            plt.text(0.5, 0.5, 'Нет примеров', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
        
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
    
    def augment_split_data(self, split_name):
        """
        Аугментирует данные для указанной выборки
        
        Args:
            split_name: Имя выборки ('train' или 'val')
        """
        if not self.enable_augmentation:
            return
        
        print(f"\n🔧 Аугментация {split_name} выборки...")
        
        # Загружаем и аугментируем данные
        X, y = self._load_split_data(split_name)
        
        if len(X) == 0:
            print(f"⚠️  Нет данных для аугментации в выборке {split_name}")
            return
        
        # Аугментируем данные
        X_aug, y_aug = self.augmentation_pipeline.augment_dataset(X, y)
        
        print(f"✅ Аугментация завершена!")
        print(f"  До аугментации: {len(X)} изображений")
        print(f"  После аугментации: {len(X_aug)} изображений")
        print(f"  Увеличение в {len(X_aug) / max(1, len(X)):.1f} раз")
        
        return X_aug, y_aug
    
    def create_numpy_dataset(self, use_augmented=True):
        """Создает numpy датасет для обучения"""
        print("\n" + "="*60)
        print("СОЗДАНИЕ ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ")
        print("="*60)
        
        # Загружаем данные
        X_train, y_train = self._load_split_data('train')
        X_val, y_val = self._load_split_data('val')
        X_test, y_test = self._load_split_data('test')
        
        # Аугментируем обучающую и валидационную выборки если нужно
        if use_augmented and self.enable_augmentation:
            print("Применение аугментации к обучающим данным...")
            X_train_aug, y_train_aug = self.augmentation_pipeline.augment_dataset(X_train, y_train)
            X_val_aug, y_val_aug = self.augmentation_pipeline.augment_dataset(X_val, y_val, 
                                                                             multiplier=max(1, self.augmentation_multiplier // 2))
            
            print(f"  Обучающая выборка: {len(X_train)} -> {len(X_train_aug)}")
            print(f"  Валидационная выборка: {len(X_val)} -> {len(X_val_aug)}")
            
            X_train, y_train = X_train_aug, y_train_aug
            X_val, y_val = X_val_aug, y_val_aug
        
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
            'augmentation_enabled': self.enable_augmentation,
            'augmentation_multiplier': self.augmentation_multiplier,
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
        if len(X_train) > 0:
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
        
        # 4. Создание numpy датасета (с аугментацией если включена)
        dataset_file = self.create_numpy_dataset(use_augmented=True)
        
        print("\n" + "="*70)
        print("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
        print("="*70)
        
        return dataset_file
    
    def show_augmentation_examples(self, num_examples=5):
        """
        Показывает примеры аугментации
        
        Args:
            num_examples: Количество примеров для показа
        """
        if not self.enable_augmentation:
            print("Аугментация отключена")
            return
        
        # Загружаем несколько изображений для демонстрации
        processed_dir = self.base_dir / 'processed'
        
        # Находим первое изображение
        sample_image = None
        for class_name in self.classes:
            class_dir = processed_dir / class_name
            image_files = list(class_dir.glob('*.png'))
            if image_files:
                sample_image_path = image_files[0]
                sample_image = Image.open(sample_image_path)
                
                # Конвертируем в numpy
                sample_array = np.array(sample_image)
                if self.grayscale and len(sample_array.shape) == 2:
                    sample_array = np.expand_dims(sample_array, axis=-1)
                sample_array = sample_array.astype(np.float32) / 255.0
                break
        
        if sample_image is None:
            print("Нет изображений для демонстрации")
            return
        
        # Создаем примеры аугментации
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Примеры аугментации изображений', fontsize=16)
        
        # Оригинальное изображение
        if self.grayscale:
            axes[0, 0].imshow(sample_array.squeeze(), cmap='gray')
        else:
            axes[0, 0].imshow(sample_array)
        axes[0, 0].set_title('Оригинал')
        axes[0, 0].axis('off')
        
        # Создаем несколько аугментированных версий
        for i in range(5):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            # Применяем аугментацию
            aug_array = self.augmentation_pipeline.apply_augmentation(sample_array)
            
            if self.grayscale:
                axes[row, col].imshow(aug_array.squeeze(), cmap='gray')
            else:
                axes[row, col].imshow(aug_array)
            
            axes[row, col].set_title(f'Аугментация {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()


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
    
    augmentation_input = input("Включить аугментацию данных? (y/n, по умолчанию: y): ").lower()
    enable_augmentation = augmentation_input != 'n'
    
    augmentation_multiplier = 10
    if enable_augmentation:
        try:
            multiplier_input = input(f"Во сколько раз увеличить датасет (по умолчанию: {augmentation_multiplier}): ")
            if multiplier_input:
                augmentation_multiplier = int(multiplier_input)
        except ValueError:
            print(f"Используется значение по умолчанию: {augmentation_multiplier}")
    
    # Создаем и запускаем подготовитель
    preparer = DataPreparer(
        dataset_name=dataset_name,
        img_size=(width, height),
        grayscale=grayscale,
        enable_augmentation=enable_augmentation,
        augmentation_multiplier=augmentation_multiplier
    )
    
    # Показываем примеры аугментации если включена
    if enable_augmentation:
        preparer.show_augmentation_examples()
    
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