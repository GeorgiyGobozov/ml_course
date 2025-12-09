# prediction_with_choice.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import json
import joblib
import pickle
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Кастомный JSON энкодер для numpy типов"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
class AdvancedImagePredictor:
    """
    Класс для предсказаний с выбором модели и визуализацией
    """
    
    def __init__(self, model_path=None):
        """
        Инициализация предсказателя
        
        Args:
            model_path (str): Путь к сохраненной модели (опционально)
        """
        self.model = None
        self.model_name = ""
        self.classes = []
        self.input_shape = None
        self.grayscale = True
        self.accuracy = 0
        self.loaded_time = None
        
        # Папки с моделями
        self.models_dir = Path("trained_models")
        self.best_model_dir = self.models_dir / "best_model"
        self.model_types_dir = self.models_dir / "model_types"
        
        # Список доступных моделей
        self.available_models = {}
        
        # Загружаем модель
        if model_path:
            self.load_model(model_path)
        else:
            self.discover_available_models()
    
    def discover_available_models(self):
        """Поиск всех доступных моделей"""
        print("🔍 Поиск доступных моделей...")
        
        # Очищаем список
        self.available_models = {
            'best': {},
            'types': {},
            'all': []
        }
        
        # 1. Ищем лучшие модели
        if self.best_model_dir.exists():
            best_models = list(self.best_model_dir.glob("*.joblib")) + \
                         list(self.best_model_dir.glob("*.pkl"))
            
            for model_path in best_models:
                model_info = self._get_model_info(model_path)
                if model_info:
                    self.available_models['best'][model_info['name']] = {
                        'path': str(model_path),  # Конвертируем в строку
                        'info': model_info
                    }
                    self.available_models['all'].append({
                        'type': 'best',
                        'name': model_info['name'],
                        'path': str(model_path),  # Конвертируем в строку
                        'info': model_info
                    })
        
        # 2. Ищем модели по типам
        if self.model_types_dir.exists():
            type_models = list(self.model_types_dir.glob("*.joblib")) + \
                         list(self.model_types_dir.glob("*.pkl"))
            
            for model_path in type_models:
                model_info = self._get_model_info(model_path)
                if model_info:
                    model_type = model_info['name'].split('_')[0] if '_' in model_info['name'] else 'other'
                    
                    if model_type not in self.available_models['types']:
                        self.available_models['types'][model_type] = []
                    
                    self.available_models['types'][model_type].append({
                        'path': str(model_path),  # Конвертируем в строку
                        'info': model_info
                    })
                    
                    self.available_models['all'].append({
                        'type': 'type',
                        'name': model_info['name'],
                        'path': str(model_path),  # Конвертируем в строку
                        'info': model_info
                    })
        
        # Сортируем по точности
        self.available_models['all'].sort(
            key=lambda x: x['info']['accuracy'] if x['info']['accuracy'] else 0, 
            reverse=True
        )
        
        print(f"✅ Найдено {len(self.available_models['all'])} моделей")
    
    def _get_model_info(self, model_path):
        """Получение информации о модели без полной загрузки"""
        try:
            model_path = Path(model_path)  # На всякий случай конвертируем в Path
            if model_path.suffix == '.joblib':
                model_data = joblib.load(model_path)
            elif model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                return None
            
            return {
                'name': model_data.get('model_name', model_path.stem),
                'accuracy': model_data.get('accuracy', 0),
                'dataset': model_data.get('dataset_name', 'Неизвестно'),
                'classes': model_data.get('classes', []),
                'timestamp': model_data.get('timestamp', 'Неизвестно')
            }
            
        except Exception as e:
            print(f"⚠️  Ошибка при получении информации о модели {model_path}: {e}")
            return None
    
    def show_available_models(self):
        """Показать список доступных моделей"""
        print("\n" + "="*70)
        print("📋 ДОСТУПНЫЕ МОДЕЛИ ДЛЯ ПРЕДСКАЗАНИЯ")
        print("="*70)
        
        if not self.available_models['all']:
            print("❌ Нет доступных моделей!")
            print("Сначала обучите модели:")
            print("python train_and_save.py --dataset ваш_датасет/dataset.npz")
            return False
        
        # Группируем по типу
        print("\n🏆 ЛУЧШИЕ МОДЕЛИ:")
        if self.available_models['best']:
            for name, model_info in self.available_models['best'].items():
                print(f"  📍 {name}")
                print(f"     Точность: {model_info['info']['accuracy']:.4f}")
                print(f"     Датасет: {model_info['info']['dataset']}")
                print(f"     Дата: {model_info['info']['timestamp']}")
                print()
        else:
            print("  (нет лучших моделей)")
        
        print("\n🎯 МОДЕЛИ ПО ТИПАМ:")
        for model_type, models in self.available_models['types'].items():
            print(f"\n  {model_type.upper()}:")
            for model in models:
                print(f"    • {model['info']['name']} - Точность: {model['info']['accuracy']:.4f}")
        
        print("\n" + "="*70)
        print("📊 ВСЕ МОДЕЛИ (отсортированы по точности):")
        print("-"*70)
        print(f"{'№':<3} {'Модель':<25} {'Точность':<10} {'Тип':<10} {'Датасет':<15}")
        print("-"*70)
        
        for i, model in enumerate(self.available_models['all']):
            print(f"{i+1:<3} {model['name']:<25} {model['info']['accuracy']:<10.4f} "
                  f"{model['type']:<10} {model['info']['dataset'][:15]:<15}")
        
        print("-"*70)
        return True
    
    def select_model_interactive(self):
        """Интерактивный выбор модели"""
        if not self.show_available_models():
            return False
        
        while True:
            try:
                choice = input("\n👉 Выберите модель (номер или название): ").strip()
                
                # Если ввели номер
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_models['all']):
                        selected_model = self.available_models['all'][idx]
                        model_path = selected_model['path']
                        break
                    else:
                        print(f"❌ Номер должен быть от 1 до {len(self.available_models['all'])}")
                
                # Если ввели название
                else:
                    # Ищем по имени
                    found_models = []
                    for model in self.available_models['all']:
                        if choice.lower() in model['name'].lower():
                            found_models.append(model)
                    
                    if len(found_models) == 1:
                        model_path = found_models[0]['path']
                        break
                    elif len(found_models) > 1:
                        print(f"\n⚠️  Найдено несколько моделей с '{choice}':")
                        for i, model in enumerate(found_models):
                            print(f"  {i+1}. {model['name']} (точность: {model['info']['accuracy']:.4f})")
                        
                        sub_choice = input("Выберите номер: ")
                        if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(found_models):
                            model_path = found_models[int(sub_choice)-1]['path']
                            break
                    else:
                        print(f"❌ Модель '{choice}' не найдена")
                        
            except KeyboardInterrupt:
                print("\n\n👋 Отмена выбора модели")
                return False
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        
        # Загружаем выбранную модель
        return self.load_model(model_path)
    
    def load_model(self, model_path):
        """Загрузка модели из файла"""
        model_path = Path(model_path)  # Конвертируем в Path для операций с файлами
        
        print(f"\n📂 Загрузка модели: {model_path}")
        
        try:
            # Пробуем разные форматы
            if model_path.suffix == '.joblib':
                model_data = joblib.load(model_path)
            elif model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                print(f"❌ Неподдерживаемый формат файла: {model_path.suffix}")
                return False
            
            # Извлекаем данные
            self.model = model_data['model']
            self.model_name = model_data.get('model_name', 'Неизвестная модель')
            self.classes = model_data.get('classes', [])
            self.input_shape = model_data.get('input_shape', (64, 64))
            self.grayscale = model_data.get('grayscale', True)
            self.accuracy = model_data.get('accuracy', 0)
            self.loaded_time = datetime.now()
            
            print(f"✅ Модель загружена успешно!")
            print(f"  Название: {self.model_name}")
            print(f"  Точность на тестовых данных: {self.accuracy:.4f}")
            print(f"  Классы ({len(self.classes)}): {', '.join(self.classes)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_image(self, image_path):
        """
        Предобработка изображения для модели
        
        Returns:
            np.array: Обработанное изображение
        """
        try:
            # Открываем изображение
            img = Image.open(image_path)
            
            # Конвертируем в нужный формат
            if self.grayscale:
                img = img.convert('L')  # Градации серого
            else:
                img = img.convert('RGB')  # Цветное
            
            # Изменяем размер
            img_resized = img.resize(self.input_shape[:2], Image.Resampling.LANCZOS)
            
            # Конвертируем в numpy array
            img_array = np.array(img_resized)
            
            # Нормализуем
            img_normalized = img_array.astype(np.float32) / 255.0
            
            # Выравниваем для классических моделей
            img_flat = img_normalized.flatten()
            
            return img_flat, img_array
            
        except Exception as e:
            print(f"❌ Ошибка при обработке изображения: {e}")
            return None, None
    
    def predict_single_image(self, image_path, show_plot=True, save_result=True):
        """
        Предсказание для одного изображения
        
        Returns:
            tuple: (предсказанный_класс, уверенность, все_вероятности)
        """
        print(f"\n🔍 Анализ изображения: {image_path}")
        
        # Предобработка
        img_flat, original_img = self.preprocess_image(image_path)
        if img_flat is None:
            return None
        
        # Предсказание
        try:
            import time
            start_time = time.time()
            
            # Для моделей с вероятностями
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([img_flat])[0]
                predicted_class_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_class_idx]
            else:
                # Для моделей без вероятностей
                predicted_class_idx = self.model.predict([img_flat])[0]
                probabilities = None
                confidence = 1.0
            
            predict_time = time.time() - start_time
            
            # Получаем название класса
            if predicted_class_idx < len(self.classes):
                predicted_class = self.classes[predicted_class_idx]
            else:
                predicted_class = f"Class_{predicted_class_idx}"
            
            print(f"📊 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ:")
            print(f"  Модель: {self.model_name}")
            print(f"  Предсказанный класс: {predicted_class}")
            print(f"  Уверенность: {confidence:.2%}")
            print(f"  Время предсказания: {predict_time:.4f} сек")
            
            if probabilities is not None:
                print(f"  Вероятности по классам:")
                for idx, prob in enumerate(probabilities):
                    if idx < len(self.classes):
                        cls_name = self.classes[idx]
                    else:
                        cls_name = f"Class_{idx}"
                    print(f"    {cls_name}: {prob:.2%}")
            
            # Визуализация
            if show_plot:
                self._visualize_prediction(image_path, original_img, predicted_class, 
                                         confidence, probabilities)
            
            # Сохранение результата
            if save_result:
                self._save_prediction_result(image_path, predicted_class, 
                                           confidence, probabilities)
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_multiple_images(self, image_dir, show_plot=True, save_results=True, 
                            max_images=10, compare_models=False):
        """
        Предсказание для нескольких изображений
    
        Returns:
            list: Результаты предсказаний
        """
        image_dir = Path(image_dir)
    
        if not image_dir.exists():
            print(f"❌ Директория не найдена: {image_dir}")
            return []
    
        # Ищем изображения РЕКУРСИВНО во всех подпапках
        print(f"🔍 Поиск изображений в {image_dir} и подпапках...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        image_files = []
    
        for ext in image_extensions:
            # Ищем в текущей директории
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
            # Ищем рекурсивно во всех подпапках
            image_files.extend(image_dir.rglob(f'*{ext}'))
            image_files.extend(image_dir.rglob(f'*{ext.upper()}'))
    
        # Убираем дубликаты
        image_files = list(set(image_files))
    
        if not image_files:
            print(f"❌ Изображения не найдены в: {image_dir}")
            print("📁 Проверьте структуру папок:")
            print("  Вариант 1: Изображения в корне папки")
            print("  Вариант 2: Изображения в подпапках (например: guitars/acoustic/, guitars/bass/)")
        
            # Показываем что есть в папке
            if image_dir.exists():
                print(f"\n📂 Содержимое папки {image_dir}:")
                items = list(image_dir.iterdir())
                if items:
                    for item in items:
                        if item.is_dir():
                            print(f"  📁 {item.name}/")
                            # Показываем содержимое подпапок
                            sub_items = list(item.glob('*'))
                            if sub_items:
                                print(f"    Файлов: {len(sub_items)}")
                        else:
                            print(f"  📄 {item.name}")
                else:
                    print("  (папка пуста)")
        
            return []
    
        print(f"✅ Найдено {len(image_files)} изображений в {image_dir}")
    
        # Показываем где найдены изображения
        print("\n📁 Изображения найдены в:")
        unique_folders = set([str(img.parent) for img in image_files])
        for folder in sorted(unique_folders)[:5]:  # Показываем первые 5 папок
            count = len([img for img in image_files if str(img.parent) == folder])
            print(f"  {folder}: {count} изображений")
    
        if len(unique_folders) > 5:
            print(f"  ... и еще {len(unique_folders) - 5} папок")
    
        results = []
    
        # Ограничиваем количество для отображения
        images_to_process = image_files[:max_images] if max_images else image_files
    
        for i, img_path in enumerate(images_to_process):
            print(f"\n[{i+1}/{len(images_to_process)}] Обработка: {img_path.relative_to(image_dir)}")
        
            result = self.predict_single_image(img_path, show_plot=False, save_result=False)
            if result:
                predicted_class, confidence, probabilities = result
                results.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'image_relative_path': str(img_path.relative_to(image_dir)),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities.tolist() if probabilities is not None else None,
                    'model_name': self.model_name,
                    'model_accuracy': self.accuracy
                })
    
        # Сохранение результатов
        if save_results and results:
            self._save_batch_results(results)
    
        # Сводная визуализация
        if show_plot and results:
            self._visualize_multiple_predictions(results)
    
        # Сравнение моделей (если нужно)
        if compare_models and len(self.available_models['all']) > 1:
            self._compare_models_on_images(images_to_process[:3])  # Первые 3 изображения
    
        return results

    def _visualize_prediction(self, image_path, original_img, predicted_class, 
                            confidence, probabilities):
        """Визуализация предсказания для одного изображения"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        fig.suptitle(
            f'РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ\n'
            f'Модель: {self.model_name} | Точность модели: {self.accuracy:.2%}',
            fontsize=14, y=1.05
        )
        
        # 1. Изображение с предсказанием
        ax1 = axes[0]
        
        if self.grayscale:
            ax1.imshow(original_img, cmap='gray')
        else:
            ax1.imshow(original_img)
        
        # Добавляем рамку с результатом
        ax1.set_title(f"Изображение: {Path(image_path).name}", fontsize=12)
        
        # Информация о предсказании
        pred_text = (
            f"Предсказано: {predicted_class}\n"
            f"Уверенность: {confidence:.2%}\n"
            f"Модель: {self.model_name}"
        )
        
        # Добавляем текстовый блок
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.05, 0.95, pred_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax1.axis('off')
        
        # 2. График вероятностей
        ax2 = axes[1]
        
        if probabilities is not None:
            # Сортируем вероятности
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_probs = probabilities[sorted_indices]
            
            # Получаем названия классов
            if len(self.classes) == len(probabilities):
                sorted_classes = [self.classes[i] for i in sorted_indices]
            else:
                sorted_classes = [f'Class {i}' for i in sorted_indices]
            
            # Ограничиваем количество отображаемых классов
            max_display = min(10, len(sorted_probs))
            sorted_probs = sorted_probs[:max_display]
            sorted_classes = sorted_classes[:max_display]
            
            # Создаем цветовую схему
            colors = ['lightgreen' if i == 0 else 'lightcoral' 
                     for i in range(len(sorted_probs))]
            
            bars = ax2.barh(sorted_classes, sorted_probs, color=colors)
            ax2.set_xlabel('Вероятность', fontsize=12)
            ax2.set_title('Вероятности по классам', fontsize=12)
            ax2.set_xlim(0, 1)
            
            # Добавляем значения на столбцы
            for bar, prob in zip(bars, sorted_probs):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.2%}', ha='left', va='center', fontsize=9)
            
            # Подсвечиваем предсказанный класс
            if predicted_class in sorted_classes:
                idx = sorted_classes.index(predicted_class)
                bars[idx].set_edgecolor('red')
                bars[idx].set_linewidth(2)
        
        else:
            ax2.text(0.5, 0.5, 'Вероятности недоступны\nдля этой модели',
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Информация о вероятностях', fontsize=12)
            ax2.axis('off')
        
        plt.tight_layout()
        
        # Сохраняем график
        result_dir = self.models_dir / "predictions_results" / "plots"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = result_dir / f"prediction_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 График сохранен: {plot_path}")
    
    def _visualize_multiple_predictions(self, results):
        """Визуализация предсказаний для нескольких изображений"""
        n_results = len(results)
        n_cols = min(3, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
        fig.suptitle(
            f'РАСПОЗНАВАНИЕ НЕСКОЛЬКИХ ИЗОБРАЖЕНИЙ\n'
            f'Модель: {self.model_name} | Точность: {self.accuracy:.2%}',
            fontsize=16, y=1.02
        )
        
        # Если одна строка, axes может быть не массивом
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, result in enumerate(results):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = axes[row, col]
            
            # Загружаем и отображаем изображение
            img = Image.open(result['image_path'])
            img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
            
            if self.grayscale:
                ax.imshow(img_resized.convert('L'), cmap='gray')
            else:
                ax.imshow(img_resized)
            
            # Добавляем информацию о предсказании
            pred_text = f"{result['predicted_class']}\n({result['confidence']:.1%})"
            
            # Цвет рамки в зависимости от уверенности
            if result['confidence'] > 0.8:
                edge_color = 'green'
                linewidth = 3
            elif result['confidence'] > 0.6:
                edge_color = 'orange'
                linewidth = 2
            else:
                edge_color = 'red'
                linewidth = 2
            
            # Добавляем рамку
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(linewidth)
            
            ax.set_title(result['image_name'], fontsize=9, pad=5)
            ax.text(0.5, -0.1, pred_text, transform=ax.transAxes, fontsize=8,
                   ha='center', va='top', color='darkblue', fontweight='bold')
            
            ax.axis('off')
        
        # Скрываем пустые subplots
        for idx in range(len(results), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Сохраняем сводный график
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = self.models_dir / "predictions_results" / f"batch_summary_{timestamp}.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Сводный график сохранен: {summary_path}")
    
    def _compare_models_on_images(self, image_paths, max_models=5):
        """Сравнение нескольких моделей на одних и тех же изображениях"""
        print("\n" + "="*70)
        print("🔄 СРАВНЕНИЕ МОДЕЛЕЙ НА ОДНИХ И ТЕХ ЖЕ ИЗОБРАЖЕНИЯХ")
        print("="*70)
        
        # Выбираем топ модели
        top_models = self.available_models['all'][:max_models]
        
        if len(top_models) < 2:
            print("⚠️  Для сравнения нужно как минимум 2 модели")
            return
        
        print(f"\n📊 Сравнение {len(top_models)} лучших моделей на {len(image_paths)} изображениях")
        
        results = {}
        
        # Для каждой модели делаем предсказания
        for model_info in top_models:
            print(f"\n🔍 Загрузка модели: {model_info['name']}...")
            
            # Временно загружаем модель
            temp_predictor = AdvancedImagePredictor(model_info['path'])
            
            model_results = []
            for img_path in image_paths:
                result = temp_predictor.predict_single_image(
                    img_path, show_plot=False, save_result=False
                )
                if result:
                    predicted_class, confidence, _ = result
                    model_results.append({
                        'image': img_path.name,
                        'predicted': predicted_class,
                        'confidence': confidence
                    })
            
            results[model_info['name']] = {
                'accuracy': model_info['info']['accuracy'],
                'predictions': model_results
            }
        
        # Визуализация сравнения
        self._visualize_model_comparison(results, image_paths)
    
    def _visualize_model_comparison(self, results, image_paths):
        """Визуализация сравнения моделей"""
        n_images = len(image_paths)
        n_models = len(results)
        
        fig, axes = plt.subplots(n_images, n_models + 1, figsize=(4*(n_models+1), 3*n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('СРАВНЕНИЕ МОДЕЛЕЙ НА ОДНИХ И ТЕХ ЖЕ ИЗОБРАЖЕНИЯХ', 
                    fontsize=14, y=1.02)
        
        # Первый столбец - оригинальные изображения
        for i, img_path in enumerate(image_paths):
            ax = axes[i, 0]
            img = Image.open(img_path)
            img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
            
            if self.grayscale:
                ax.imshow(img_resized.convert('L'), cmap='gray')
            else:
                ax.imshow(img_resized)
            
            ax.set_title(f"Изображение {i+1}", fontsize=10)
            ax.axis('off')
            
            # Добавляем названия моделей в заголовки
            if i == 0:
                axes[0, 0].set_title("Оригинал", fontsize=11)
        
        # Предсказания моделей
        model_names = list(results.keys())
        
        for model_idx, model_name in enumerate(model_names):
            model_data = results[model_name]
            
            for img_idx in range(n_images):
                ax = axes[img_idx, model_idx + 1]
                
                if img_idx < len(model_data['predictions']):
                    pred = model_data['predictions'][img_idx]
                    
                    # Отображаем предсказание
                    ax.text(0.5, 0.6, pred['predicted'], 
                           ha='center', va='center', fontsize=11, fontweight='bold')
                    
                    ax.text(0.5, 0.3, f"Уверенность: {pred['confidence']:.1%}",
                           ha='center', va='center', fontsize=9)
                    
                    # Цвет фона в зависимости от уверенности
                    if pred['confidence'] > 0.8:
                        ax.set_facecolor('#e8f5e9')  # светло-зеленый
                    elif pred['confidence'] > 0.6:
                        ax.set_facecolor('#fff3e0')  # светло-оранжевый
                    else:
                        ax.set_facecolor('#ffebee')  # светло-красный
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Заголовок с точностью модели
                if img_idx == 0:
                    ax.set_title(f"{model_name}\n(Точность: {model_data['accuracy']:.2%})", 
                                fontsize=10, pad=10)
        
        plt.tight_layout()
        
        # Сохраняем график сравнения
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        compare_path = self.models_dir / "predictions_results" / f"model_comparison_{timestamp}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 График сравнения сохранен: {compare_path}")
    
    def _save_prediction_result(self, image_path, predicted_class, confidence, probabilities):
        """Сохранение результата предсказания"""
        result_dir = self.models_dir / "predictions_results" / "single"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = result_dir / f"prediction_{timestamp}.json"
        
        result_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': {
                'name': self.model_name,
                'accuracy': self.accuracy
            },
            'image': {
                'path': str(image_path),  # Конвертируем в строку
                'name': Path(image_path).name
            },
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist() if probabilities is not None else None
            },
            'classes': self.classes
        }
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Результат сохранен: {result_file}")
            
            # Также сохраняем в текстовом формате
            txt_file = result_dir / f"prediction_{timestamp}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ ИЗОБРАЖЕНИЯ\n")
                f.write("="*60 + "\n\n")
                f.write(f"Дата и время: {result_data['timestamp']}\n")
                f.write(f"Модель: {self.model_name}\n")
                f.write(f"Точность модели: {self.accuracy:.4f}\n")
                f.write(f"Изображение: {image_path}\n")
                f.write(f"Предсказанный класс: {predicted_class}\n")
                f.write(f"Уверенность: {confidence:.2%}\n\n")
                
                if probabilities is not None:
                    f.write("Вероятности по классам:\n")
                    for idx, prob in enumerate(probabilities):
                        if idx < len(self.classes):
                            cls_name = self.classes[idx]
                        else:
                            cls_name = f"Class_{idx}"
                        f.write(f"  {cls_name}: {prob:.2%}\n")
        except Exception as e:
            print(f"❌ Ошибка при сохранении результата: {e}")
    
    def _save_batch_results(self, results):
        """Сохранение результатов пакетной обработки"""
        result_dir = self.models_dir / "predictions_results" / "batch"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Сохраняем в JSON
        json_file = result_dir / f"batch_results_{timestamp}.json"
        
        results_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': {
                'name': self.model_name,
                'accuracy': self.accuracy
            },
            'total_images': len(results),
            'predictions': results
        }
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            # Сохраняем в CSV
            import pandas as pd
            csv_data = []
            for result in results:
                csv_data.append({
                    'Изображение': result['image_name'],
                    'Предсказанный класс': result['predicted_class'],
                    'Уверенность': f"{result['confidence']:.2%}",
                    'Модель': result['model_name'],
                    'Точность модели': f"{result['model_accuracy']:.4f}"
                })
            
            df = pd.DataFrame(csv_data)
            csv_file = result_dir / f"batch_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # Выводим таблицу
            print(f"\n📋 СВОДНАЯ ТАБЛИЦА ПРЕДСКАЗАНИЙ:")
            print("="*80)
            print(df.to_string(index=False))
            print("="*80)
            
            print(f"\n💾 Результаты сохранены:")
            print(f"  JSON: {json_file}")
            print(f"  CSV: {csv_file}")
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении пакетных результатов: {e}")
            import traceback
            traceback.print_exc()
    
    def show_model_info(self):
        """Показать информацию о загруженной модели"""
        if self.model is None:
            print("❌ Модель не загружена!")
            return
        
        print("\n" + "="*70)
        print("📊 ИНФОРМАЦИЯ О ЗАГРУЖЕННОЙ МОДЕЛИ")
        print("="*70)
        print(f"Название модели: {self.model_name}")
        print(f"Тип модели: {type(self.model).__name__}")
        print(f"Точность на тестовых данных: {self.accuracy:.4f}")
        print(f"Дата загрузки: {self.loaded_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Количество классов: {len(self.classes)}")
        print(f"Классы: {', '.join(self.classes)}")
        print(f"Размер входных изображений: {self.input_shape}")
        print(f"Градации серого: {'Да' if self.grayscale else 'Нет'}")
        
        # Дополнительная информация
        print("\n⚙️  ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        
        if hasattr(self.model, 'n_features_in_'):
            print(f"  Количество признаков: {self.model.n_features_in_}")
        
        if hasattr(self.model, 'n_iter_'):
            print(f"  Количество итераций обучения: {self.model.n_iter_}")
        
        if hasattr(self.model, 'n_estimators'):
            print(f"  Количество деревьев: {self.model.n_estimators}")
        
        if hasattr(self.model, 'kernel'):
            print(f"  Тип ядра SVM: {self.model.kernel}")


# Утилитарные функции
import time

def select_image_interactive():
    """Интерактивный выбор изображения"""
    print("\n📁 ВЫБОР ИЗОБРАЖЕНИЯ ДЛЯ АНАЛИЗА")
    print("="*50)
    
    # Предлагаем варианты
    print("1. Ввести путь к изображению")
    print("2. Выбрать из текущей директории")
    print("3. Использовать тестовое изображение")
    
    choice = input("\n👉 Выберите вариант (1-3): ")
    
    if choice == '1':
        image_path = input("Введите полный путь к изображению: ")
        return Path(image_path)
    
    elif choice == '2':
        # Ищем изображения в текущей директории
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path('.').glob(f'*{ext}'))
            image_files.extend(Path('.').glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("❌ В текущей директории нет изображений")
            return None
        
        print("\n📷 Найдены изображения:")
        for i, img_file in enumerate(image_files):
            print(f"  {i+1}. {img_file.name}")
        
        img_choice = input("\n👉 Выберите изображение (номер): ")
        
        if img_choice.isdigit() and 1 <= int(img_choice) <= len(image_files):
            return image_files[int(img_choice) - 1]
        else:
            print("❌ Неверный выбор")
            return None
    
    elif choice == '3':
        # Используем тестовое изображение
        test_images = list(Path('.').glob('test_*.jpg')) + \
                     list(Path('.').glob('test_*.png'))
        
        if test_images:
            return test_images[0]
        else:
            print("❌ Тестовые изображения не найдены")
            return None
    
    return None


def select_folder_interactive():
    """Интерактивный выбор папки"""
    print("\n📁 ВЫБОР ПАПКИ С ИЗОБРАЖЕНИЯМИ")
    print("="*50)
    
    print("1. Ввести путь к папке")
    print("2. Использовать текущую директорию")
    print("3. Использовать тестовую папку")
    
    choice = input("\n👉 Выберите вариант (1-3): ")
    
    if choice == '1':
        folder_path = input("Введите путь к папке: ")
        return Path(folder_path)
    
    elif choice == '2':
        return Path('.')
    
    elif choice == '3':
        test_folders = list(Path('.').glob('test_*')) + \
                      list(Path('.').glob('images'))
        
        if test_folders:
            for folder in test_folders:
                if folder.is_dir():
                    return folder
        
        print("❌ Тестовые папки не найдены")
        return Path('.')
    
    return None


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Предсказание классов изображений с выбором модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python prediction_with_choice.py --select-model --image photo.jpg
  python prediction_with_choice.py --list-models
  python prediction_with_choice.py --model path/to/model.joblib --folder images/
        '''
    )
    
    parser.add_argument('--model', type=str, 
                       help='Путь к конкретной модели')
    
    parser.add_argument('--select-model', action='store_true',
                       help='Интерактивный выбор модели из доступных')
    
    parser.add_argument('--list-models', action='store_true',
                       help='Показать список доступных моделей')
    
    parser.add_argument('--image', type=str,
                       help='Путь к одному изображению')
    
    parser.add_argument('--folder', type=str,
                       help='Путь к папке с изображениями')
    
    parser.add_argument('--max-images', type=int, default=10,
                       help='Максимальное количество изображений')
    
    parser.add_argument('--compare', action='store_true',
                       help='Сравнить несколько моделей')
    
    parser.add_argument('--info', action='store_true',
                       help='Показать информацию о загруженной модели')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🖼️  СИСТЕМА РАСПОЗНАВАНИЯ ИЗОБРАЖЕНИЙ С ВЫБОРОМ МОДЕЛИ")
    print("="*70)
    
    # Создаем предсказатель
    predictor = AdvancedImagePredictor(args.model if args.model else None)
    
    # Показать список моделей
    if args.list_models:
        predictor.show_available_models()
        return
    
    # Показать информацию о модели
    if args.info:
        predictor.show_model_info()
        return
    
    # Если модель не загружена, предлагаем выбрать
    if predictor.model is None or args.select_model:
        if not predictor.select_model_interactive():
            print("❌ Не удалось загрузить модель")
            return
    
    # Обработка одного изображения
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Изображение не найдено: {image_path}")
            return
        
        predictor.predict_single_image(image_path, show_plot=True, save_result=True)
    
    # Обработка папки с изображениями
    elif args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"❌ Папка не найдена: {folder_path}")
            return
        
        results = predictor.predict_multiple_images(
            folder_path, 
            show_plot=True, 
            save_results=True,
            max_images=args.max_images,
            compare_models=args.compare
        )
        
        if results:
            print(f"\n✅ Обработано {len(results)} изображений")
    
    else:
        # Интерактивный режим
        print("\n📋 ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("="*50)
        
        print("1. 🔍 Анализ одного изображения")
        print("2. 📁 Анализ папки с изображениями")
        print("3. 📊 Информация о модели")
        print("4. 🔄 Сравнение моделей")
        print("5. ❌ Выход")
        
        choice = input("\n👉 Выберите действие (1-5): ")
        
        if choice == '1':
            image_path = select_image_interactive()
            if image_path and image_path.exists():
                predictor.predict_single_image(image_path, show_plot=True, save_result=True)
            else:
                print("❌ Не удалось выбрать изображение")
        
        elif choice == '2':
            folder_path = select_folder_interactive()
            if folder_path and folder_path.exists():
                max_images = input("Максимальное количество изображений (по умолчанию 10): ")
                max_images = int(max_images) if max_images and max_images.isdigit() else 10
                
                results = predictor.predict_multiple_images(
                    folder_path, 
                    show_plot=True, 
                    save_results=True,
                    max_images=max_images
                )
                
                if results:
                    print(f"\n✅ Обработано {len(results)} изображений")
            else:
                print("❌ Не удалось выбрать папку")
        
        elif choice == '3':
            predictor.show_model_info()
        
        elif choice == '4':
            if len(predictor.available_models['all']) > 1:
                # Сравнение на тестовых изображениях
                test_image = select_image_interactive()
                if test_image:
                    predictor._compare_models_on_images([test_image])
            else:
                print("❌ Для сравнения нужно как минимум 2 модели")
        
        elif choice == '5':
            print("\n👋 До свидания!")
            return
        
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем.")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")