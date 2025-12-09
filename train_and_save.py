# train_and_save.py - обучение и сохранение моделей
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import time
import json
import joblib
import pickle
import argparse
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Импортируем модели sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class ModelTrainer:
    """
    Класс для обучения и сохранения моделей
    """
    
    def __init__(self, dataset_path):
        """
        Инициализация тренера моделей
        
        Args:
            dataset_path (str): Путь к файлу dataset.npz
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = ""
        self.classes = []
        self.input_shape = None
        self.grayscale = True
        self.results = {}
        self.models_dir = Path("trained_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Создаем поддиректории для моделей
        self.model_types_dir = self.models_dir / "model_types"
        self.model_types_dir.mkdir(exist_ok=True)
        
        self.best_model_dir = self.models_dir / "best_model"
        self.best_model_dir.mkdir(exist_ok=True)
        
        # Загружаем данные
        self._load_dataset()
        
        # Подготавливаем данные
        self._prepare_data()
    
    def _load_dataset(self):
        """Загрузка датасета из файла"""
        print(f"📂 Загрузка датасета: {self.dataset_path}")
        
        try:
            data = np.load(self.dataset_path)
            
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_val = data['X_val']
            self.y_val = data['y_val']
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            
            # Загружаем метаданные
            metadata_path = self.dataset_path.parent / 'metadata' / 'dataset_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.dataset_name = metadata.get('dataset_name', 'Неизвестный датасет')
                self.classes = metadata.get('classes', [])
                self.input_shape = metadata.get('input_shape', self.X_train[0].shape)
                self.grayscale = metadata.get('grayscale', True)
                
                print(f"📊 ИНФОРМАЦИЯ О ДАТАСЕТЕ:")
                print(f"  Название: {self.dataset_name}")
                print(f"  Классы ({len(self.classes)}): {', '.join(self.classes)}")
                print(f"  Обучающая выборка: {self.X_train.shape[0]} изображений")
                print(f"  Тестовая выборка: {self.X_test.shape[0]} изображений")
                print(f"  Размер изображения: {self.input_shape}")
                
            else:
                print("⚠️  Метаданные не найдены")
                self.classes = [f'Class_{i}' for i in range(len(np.unique(self.y_train)))]
                
        except Exception as e:
            print(f"❌ Ошибка при загрузке датасета: {e}")
            sys.exit(1)
    
    def _prepare_data(self):
        """Подготовка данных для обучения"""
        print("\n🔧 Подготовка данных для обучения...")
        
        # Выравниваем изображения для классических моделей
        n_train = self.X_train.shape[0]
        n_val = self.X_val.shape[0]
        n_test = self.X_test.shape[0]
        
        self.X_train_flat = self.X_train.reshape(n_train, -1)
        self.X_val_flat = self.X_val.reshape(n_val, -1)
        self.X_test_flat = self.X_test.reshape(n_test, -1)
        
        # Объединяем обучающую и валидационную выборки
        self.X_train_full = np.vstack([self.X_train_flat, self.X_val_flat])
        self.y_train_full = np.concatenate([self.y_train, self.y_val])
        
        print(f"✅ Данные подготовлены:")
        print(f"  Всего обучающих изображений: {self.X_train_full.shape[0]}")
        print(f"  Тестовых изображений: {self.X_test_flat.shape[0]}")
        print(f"  Размер признаков: {self.X_train_full.shape[1]}")
    
    def train_model(self, model, model_name, save_model=True):
        """
        Обучение и сохранение модели
        
        Returns:
            dict: Результаты обучения
        """
        print(f"\n🎯 Обучение модели: {model_name}")
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        # Предсказания
        start_time = time.time()
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
        # Расчет метрик
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'predict_time': predict_time,
            'predictions': y_pred,
            'model_name': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"  Точность: {accuracy:.4f}")
        print(f"  F1-мера: {f1:.4f}")
        print(f"  Время обучения: {train_time:.2f} сек")
        
        # Сохранение модели
        if save_model:
            self._save_model(model, model_name, results)
        
        self.results[model_name] = results
        return results
    
    def _save_model(self, model, model_name, results):
        """Сохранение модели в файл"""
        # Создаем имя файла
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Сохраняем модель с помощью joblib
        model_path = self.model_types_dir / f"{filename}.joblib"
        
        # Создаем словарь с моделью и метаданными
        model_data = {
            'model': model,
            'model_name': model_name,
            'dataset_name': self.dataset_name,
            'classes': self.classes,
            'input_shape': self.input_shape,
            'grayscale': self.grayscale,
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'train_time': results['train_time'],
            'timestamp': results['timestamp']
        }
        
        joblib.dump(model_data, model_path)
        
        # Также сохраняем в pickle для совместимости
        pickle_path = self.model_types_dir / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  💾 Модель сохранена: {model_path}")
        return model_path
    
    def train_all_models(self):
        """Обучение всех моделей"""
        print("\n" + "="*70)
        print("🚀 ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛЕЙ")
        print("="*70)
        
        # Определяем все модели для обучения
        models = [
            ("Logistic_Regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("SVM_Linear", SVC(kernel='linear', random_state=42, probability=True)),
            ("SVM_RBF", SVC(kernel='rbf', random_state=42, probability=True)),
            ("Random_Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("KNN", KNeighborsClassifier(n_neighbors=5)),
            ("Decision_Tree", DecisionTreeClassifier(random_state=42)),
            ("Naive_Bayes", GaussianNB()),
            ("MLP", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
        ]
        
        for name, model in models:
            try:
                self.train_model(model, name, save_model=True)
            except Exception as e:
                print(f"❌ Ошибка при обучении {name}: {e}")
        
        # Определяем лучшую модель
        self._select_best_model()
        
        # Создаем сводный отчет
        self._create_summary_report()
        
        # Визуализация результатов
        self._visualize_results()
        
        print(f"\n✅ Все модели обучены и сохранены в папке: {self.models_dir}")
    
    def _select_best_model(self):
        """Выбор и сохранение лучшей модели"""
        if not self.results:
            print("⚠️  Нет результатов для выбора лучшей модели")
            return
        
        # Находим модель с максимальной точностью
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_result = self.results[best_model_name]
        
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
        print(f"  Точность: {best_result['accuracy']:.4f}")
        print(f"  F1-мера: {best_result['f1_score']:.4f}")
        
        # Сохраняем лучшую модель в отдельную папку
        best_model_data = {
            'model': best_result['model'],
            'model_name': best_model_name,
            'dataset_name': self.dataset_name,
            'classes': self.classes,
            'input_shape': self.input_shape,
            'grayscale': self.grayscale,
            'accuracy': best_result['accuracy'],
            'f1_score': best_result['f1_score'],
            'train_time': best_result['train_time'],
            'timestamp': best_result['timestamp'],
            'all_results': self.results
        }
        
        # Сохраняем разными способами для максимальной совместимости
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Joblib
        joblib_path = self.best_model_dir / f"best_model_{timestamp}.joblib"
        joblib.dump(best_model_data, joblib_path)
        
        # 2. Pickle
        pickle_path = self.best_model_dir / f"best_model_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(best_model_data, f)
        
        # 3. Сохраняем как "latest" для удобства
        latest_path = self.best_model_dir / "best_model_latest.joblib"
        joblib.dump(best_model_data, latest_path)
        
        print(f"💾 Лучшая модель сохранена:")
        print(f"  Основная: {joblib_path}")
        print(f"  Последняя: {latest_path}")
        
        # Сохраняем информацию о лучшей модели в текстовый файл
        info_path = self.best_model_dir / "best_model_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ИНФОРМАЦИЯ О ЛУЧШЕЙ МОДЕЛИ\n")
            f.write("="*60 + "\n\n")
            f.write(f"Название модели: {best_model_name}\n")
            f.write(f"Датасет: {self.dataset_name}\n")
            f.write(f"Дата обучения: {best_result['timestamp']}\n")
            f.write(f"Точность: {best_result['accuracy']:.4f}\n")
            f.write(f"F1-мера: {best_result['f1_score']:.4f}\n")
            f.write(f"Время обучения: {best_result['train_time']:.2f} сек\n")
            f.write(f"Классы: {', '.join(self.classes)}\n")
            f.write(f"Размер изображений: {self.input_shape}\n")
        
        self.best_model_info = best_model_data
    
    def _create_summary_report(self):
        """Создание сводного отчета"""
        if not self.results:
            return
        
        # Создаем отчет в JSON
        report_data = {
            'dataset_name': self.dataset_name,
            'classes': self.classes,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models': {}
        }
        
        for model_name, res in self.results.items():
            report_data['models'][model_name] = {
                'accuracy': float(res['accuracy']),
                'f1_score': float(res['f1_score']),
                'precision': float(res['precision']),
                'recall': float(res['recall']),
                'train_time': float(res['train_time']),
                'predict_time': float(res['predict_time'])
            }
        
        # Определяем лучшую модель
        if self.results:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            report_data['best_model'] = {
                'name': best_model,
                'accuracy': float(self.results[best_model]['accuracy']),
                'f1_score': float(self.results[best_model]['f1_score'])
            }
        
        # Сохраняем отчет
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Сводный отчет сохранен: {report_path}")
        
        # Также сохраняем в CSV
        import pandas as pd
        
        csv_data = []
        for model_name, res in self.results.items():
            csv_data.append({
                'Model': model_name,
                'Accuracy': res['accuracy'],
                'F1_Score': res['f1_score'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'Train_Time_s': res['train_time'],
                'Predict_Time_s': res['predict_time']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.models_dir / "models_comparison.csv", index=False)
        print(f"📋 Таблица сравнения: {self.models_dir / 'models_comparison.csv'}")
    
    def _visualize_results(self):
        """Визуализация результатов обучения"""
        if len(self.results) < 2:
            return
        
        plt.figure(figsize=(14, 8))
        
        # 1. Сравнение точности
        plt.subplot(2, 3, 1)
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        colors = plt.cm.Set3(np.arange(len(model_names)) / len(model_names))
        bars = plt.bar(model_names, accuracies, color=colors)
        plt.title('Сравнение точности моделей', fontsize=14)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Сравнение F1-меры
        plt.subplot(2, 3, 2)
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        
        bars = plt.bar(model_names, f1_scores, color=colors)
        plt.title('Сравнение F1-меры', fontsize=14)
        plt.ylabel('F1-Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Время обучения
        plt.subplot(2, 3, 3)
        train_times = [self.results[name]['train_time'] for name in model_names]
        
        plt.bar(model_names, train_times, color='lightcoral')
        plt.title('Время обучения моделей', fontsize=14)
        plt.ylabel('Время (сек)')
        plt.xticks(rotation=45)
        
        # 4. Матрица ошибок лучшей модели
        plt.subplot(2, 3, 4)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_predictions = self.results[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, best_predictions)
        
        if self.classes and len(self.classes) == cm.shape[0]:
            tick_labels = self.classes
        else:
            tick_labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=tick_labels,
                   yticklabels=tick_labels)
        plt.title(f'Матрица ошибок: {best_model_name}', fontsize=14)
        plt.xlabel('Предсказанные метки')
        plt.ylabel('Истинные метки')
        
        # 5. Сравнение точности и F1
        plt.subplot(2, 3, 5)
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightgreen')
        plt.title('Сравнение Accuracy и F1-Score', fontsize=14)
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        
        # 6. Время предсказания
        plt.subplot(2, 3, 6)
        predict_times = [self.results[name]['predict_time'] for name in model_names]
        
        plt.bar(model_names, predict_times, color='lightblue')
        plt.title('Время предсказания', fontsize=14)
        plt.ylabel('Время (сек)')
        plt.xticks(rotation=45)
        
        plt.suptitle(f'Результаты обучения моделей на датасете "{self.dataset_name}"', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Сохраняем график
        plot_path = self.models_dir / "models_comparison_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📈 График сравнения сохранен: {plot_path}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обучение и сохранение моделей для распознавания образов')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Путь к файлу dataset.npz')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🤖 СИСТЕМА ОБУЧЕНИЯ И СОХРАНЕНИЯ МОДЕЛЕЙ")
    print("="*70)
    
    # Создаем тренер
    trainer = ModelTrainer(args.dataset)
    
    # Обучаем и сохраняем все модели
    trainer.train_all_models()
    
    print("\n" + "="*70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*70)
    print("\n📁 Структура сохраненных моделей:")
    print("  trained_models/")
    print("  ├── model_types/      # Все обученные модели")
    print("  ├── best_model/       # Лучшая модель")
    print("  ├── training_report.json   # Отчет")
    print("  └── models_comparison.csv  # Таблица сравнения")
    print("\n🏆 Лучшая модель готова к использованию в prediction.py!")


if __name__ == "__main__":
    main()