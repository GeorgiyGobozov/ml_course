# model_comparison_detailed.py - версия с подробным выводом для каждой модели
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score
import time
import json
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Импортируем модели sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import label_binarize

class DetailedModelComparator:
    """
    Компаратор моделей с подробным выводом результатов для каждой модели
    """
    
    def __init__(self, dataset_path):
        from pathlib import Path
        self.dataset_path = Path(dataset_path)
        self.dataset_name = ""
        self.classes = []
        self.input_shape = None
        self.grayscale = True
        self.results = {}
        self.detailed_reports = {}
        
        # Загружаем данные
        self._load_dataset()
        
        # Подготавливаем данные
        self._prepare_data()
    
    def _load_dataset(self):
        """Загрузка датасета из файла"""
        print(f"📂 Загрузка датасета: {self.dataset_path}")
        
        try:
            # Загружаем данные
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
                print(f"  Количество классов: {len(self.classes)}")
                print(f"  Классы: {', '.join(self.classes)}")
                print(f"  Обучающая выборка: {self.X_train.shape[0]} изображений")
                print(f"  Валидационная выборка: {self.X_val.shape[0]} изображений")
                print(f"  Тестовая выборка: {self.X_test.shape[0]} изображений")
                print(f"  Размер изображения: {self.input_shape}")
                print(f"  Градации серого: {'Да' if self.grayscale else 'Нет'}")
                
                # Распределение классов
                print(f"\n📈 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
                for split_name, X_split, y_split in [
                    ('Обучающая', self.X_train, self.y_train),
                    ('Валидационная', self.X_val, self.y_val),
                    ('Тестовая', self.X_test, self.y_test)
                ]:
                    unique, counts = np.unique(y_split, return_counts=True)
                    print(f"  {split_name} выборка:")
                    for cls_idx, count in zip(unique, counts):
                        cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else f'Class_{cls_idx}'
                        percentage = (count / len(y_split)) * 100
                        print(f"    {cls_name}: {count} изображений ({percentage:.1f}%)")
                
            else:
                print("⚠️  Метаданные не найдены")
                # Автоматически определяем классы
                self.classes = [f'Class_{i}' for i in range(len(np.unique(self.y_train)))]
                
        except Exception as e:
            print(f"❌ Ошибка при загрузке датасета: {e}")
            sys.exit(1)
    
    def _prepare_data(self):
        """Подготовка данных для обучения"""
        print("\n🔧 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        
        # Преобразуем данные для sklearn моделей
        n_train = self.X_train.shape[0]
        n_val = self.X_val.shape[0]
        n_test = self.X_test.shape[0]
        
        # Выравниваем изображения
        self.X_train_flat = self.X_train.reshape(n_train, -1)
        self.X_val_flat = self.X_val.reshape(n_val, -1)
        self.X_test_flat = self.X_test.reshape(n_test, -1)
        
        # Объединяем обучающую и валидационную выборки
        self.X_train_full = np.vstack([self.X_train_flat, self.X_val_flat])
        self.y_train_full = np.concatenate([self.y_train, self.y_val])
        
        print(f"✅ Данные подготовлены:")
        print(f"  Размер обучающих данных: {self.X_train_flat.shape}")
        print(f"  Размер тестовых данных: {self.X_test_flat.shape}")
        print(f"  Всего признаков (features): {self.X_train_flat.shape[1]}")
        print(f"  Диапазон значений пикселей: [{self.X_train.min():.3f}, {self.X_train.max():.3f}]")
    
    def _print_model_header(self, model_name):
        """Печать заголовка для модели"""
        print("\n" + "="*70)
        print(f"МОДЕЛЬ: {model_name}")
        print("="*70)
    
    def _print_model_results(self, model_name, results, y_pred):
        """Печать подробных результатов для модели"""
        print(f"\n📊 РЕЗУЛЬТАТЫ ДЛЯ {model_name}:")
        print(f"  Точность (Accuracy): {results['accuracy']:.4f}")
        print(f"  F1-мера: {results['f1_score']:.4f}")
        print(f"  Точность (Precision): {results['precision']:.4f}")
        print(f"  Полнота (Recall): {results['recall']:.4f}")
        print(f"  Время обучения: {results['train_time']:.2f} сек")
        print(f"  Время предсказания: {results['predict_time']:.4f} сек")
        
        # Детальный отчет по классам
        if len(self.classes) > 0:
            print(f"\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
            report = classification_report(self.y_test, y_pred, 
                                          target_names=self.classes,
                                          output_dict=True)
            
            # Сохраняем отчет для последующего использования
            self.detailed_reports[model_name] = report
            
            # Выводим таблицу
            print(f"{'Класс':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-"*55)
            
            for cls in self.classes:
                if cls in report:
                    cls_report = report[cls]
                    print(f"{cls:<15} {cls_report['precision']:<10.3f} "
                          f"{cls_report['recall']:<10.3f} {cls_report['f1-score']:<10.3f} "
                          f"{int(cls_report['support']):<10}")
            
            print("-"*55)
            print(f"{'Среднее/Итого':<15} {report['weighted avg']['precision']:<10.3f} "
                  f"{report['weighted avg']['recall']:<10.3f} "
                  f"{report['weighted avg']['f1-score']:<10.3f} "
                  f"{int(report['weighted avg']['support']):<10}")
        
        # Примеры правильных и неправильных предсказаний
        self._show_prediction_examples(model_name, y_pred, num_examples=3)
    
    def _show_prediction_examples(self, model_name, y_pred, num_examples=3):
        """Показать примеры правильных и неправильных предсказаний"""
        print(f"\n🔍 ПРИМЕРЫ ПРЕДСКАЗАНИЙ:")
        
        # Находим правильные и неправильные предсказания
        correct_indices = np.where(y_pred == self.y_test)[0]
        incorrect_indices = np.where(y_pred != self.y_test)[0]
        
        if len(correct_indices) > 0 and len(incorrect_indices) > 0:
            # Выбираем случайные примеры
            np.random.seed(42)
            correct_samples = np.random.choice(correct_indices, 
                                              min(num_examples, len(correct_indices)), 
                                              replace=False)
            incorrect_samples = np.random.choice(incorrect_indices, 
                                                min(num_examples, len(incorrect_indices)), 
                                                replace=False)
            
            print("  Правильные предсказания:")
            for idx in correct_samples:
                true_label = self.classes[self.y_test[idx]] if self.y_test[idx] < len(self.classes) else f'Class_{self.y_test[idx]}'
                pred_label = self.classes[y_pred[idx]] if y_pred[idx] < len(self.classes) else f'Class_{y_pred[idx]}'
                print(f"    Изображение {idx}: Истинное = {true_label}, Предсказанное = {pred_label}")
            
            print("\n  Ошибочные предсказания:")
            for idx in incorrect_samples:
                true_label = self.classes[self.y_test[idx]] if self.y_test[idx] < len(self.classes) else f'Class_{self.y_test[idx]}'
                pred_label = self.classes[y_pred[idx]] if y_pred[idx] < len(self.classes) else f'Class_{y_pred[idx]}'
                print(f"    Изображение {idx}: Истинное = {true_label}, Предсказанное = {pred_label}")
        else:
            print("  Не удалось найти примеры предсказаний")
    
    def evaluate_logistic_regression(self):
        """Оценка модели логистической регрессии"""
        model_name = "Logistic Regression"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: Логистическая регрессия")
        print("  Максимальное количество итераций: 1000")
        print("  Регуляризация: L2")
        
        model = LogisticRegression(max_iter=1000, random_state=42, verbose=0)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        # Получаем вероятность для каждого класса
        start_time = time.time()
        y_pred_proba = model.predict_proba(self.X_test_flat)
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
            'predictions_proba': y_pred_proba
        }
        
        self._print_model_results(model_name, results, y_pred)
        self.results[model_name] = results
        
        return results
    
    def evaluate_svm_linear(self):
        """Оценка SVM с линейным ядром"""
        model_name = "SVM (Linear Kernel)"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: Метод опорных векторов")
        print("  Ядро: линейное")
        print("  Параметр регуляризации C: 1.0")
        
        model = SVC(kernel='linear', random_state=42, probability=True)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_proba = model.predict_proba(self.X_test_flat)
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
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
            'predictions_proba': y_pred_proba
        }
        
        self._print_model_results(model_name, results, y_pred)
        self.results[model_name] = results
        
        return results
    
    def evaluate_svm_rbf(self):
        """Оценка SVM с RBF ядром"""
        model_name = "SVM (RBF Kernel)"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: Метод опорных векторов")
        print("  Ядро: радиально-базисная функция (RBF)")
        print("  Параметр регуляризации C: 1.0")
        print("  Параметр gamma: 'scale'")
        
        model = SVC(kernel='rbf', random_state=42, probability=True)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_proba = model.predict_proba(self.X_test_flat)
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
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
            'predictions_proba': y_pred_proba
        }
        
        self._print_model_results(model_name, results, y_pred)
        self.results[model_name] = results
        
        return results
    
    def evaluate_random_forest(self):
        """Оценка случайного леса"""
        model_name = "Random Forest"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: Случайный лес")
        print("  Количество деревьев: 100")
        print("  Максимальная глубина: не ограничена")
        print("  Критерий разделения: критерий Джини")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_proba = model.predict_proba(self.X_test_flat)
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Важность признаков
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-5:][::-1]
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'predict_time': predict_time,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'feature_importance': feature_importance
        }
        
        self._print_model_results(model_name, results, y_pred)
        
        # Дополнительная информация для Random Forest
        print(f"\n🌲 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        print(f"  Средняя глубина деревьев: {np.mean([tree.tree_.max_depth for tree in model.estimators_]):.1f}")
        print(f"  Топ-5 важных признаков: {top_features[:5]}")
        
        self.results[model_name] = results
        
        return results
    
    def evaluate_knn(self):
        """Оценка K-ближайших соседей"""
        model_name = "K-Nearest Neighbors"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: K-ближайших соседей")
        print("  Количество соседей: 5")
        print("  Метрика расстояния: Евклидово расстояние")
        print("  Веса: равные")
        
        model = KNeighborsClassifier(n_neighbors=5)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
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
            'predictions': y_pred
        }
        
        self._print_model_results(model_name, results, y_pred)
        self.results[model_name] = results
        
        return results
    
    def evaluate_decision_tree(self):
        """Оценка решающего дерева"""
        model_name = "Decision Tree"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: Решающее дерево")
        print("  Критерий разделения: критерий Джини")
        print("  Максимальная глубина: не ограничена")
        print("  Минимальное количество образцов для разделения: 2")
        
        model = DecisionTreeClassifier(random_state=42)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
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
            'predictions': y_pred
        }
        
        self._print_model_results(model_name, results, y_pred)
        
        # Дополнительная информация для Decision Tree
        print(f"\n🌳 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        print(f"  Глубина дерева: {model.get_depth()}")
        print(f"  Количество листьев: {model.get_n_leaves()}")
        
        self.results[model_name] = results
        
        return results
    
    def evaluate_naive_bayes(self):
        """Оценка наивного Байеса"""
        model_name = "Gaussian Naive Bayes"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Алгоритм: Наивный Байес (Гауссовский)")
        print("  Предположение: признаки независимы и имеют нормальное распределение")
        
        model = GaussianNB()
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_proba = model.predict_proba(self.X_test_flat)
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
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
            'predictions_proba': y_pred_proba
        }
        
        self._print_model_results(model_name, results, y_pred)
        self.results[model_name] = results
        
        return results
    
    def evaluate_mlp(self):
        """Оценка многослойного перцептрона"""
        model_name = "Neural Network (MLP)"
        self._print_model_header(model_name)
        
        print("Параметры модели:")
        print("  Архитектура: Многослойный перцептрон")
        print("  Скрытые слои: (128, 64) нейронов")
        print("  Функция активации: ReLU")
        print("  Оптимизатор: Adam")
        print("  Максимальное количество эпох: 500")
        
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, 
                             random_state=42, verbose=False)
        
        start_time = time.time()
        model.fit(self.X_train_full, self.y_train_full)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_proba = model.predict_proba(self.X_test_flat)
        y_pred = model.predict(self.X_test_flat)
        predict_time = time.time() - start_time
        
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
            'predictions_proba': y_pred_proba,
            'n_iter': model.n_iter_,
            'loss_curve': model.loss_curve_
        }
        
        self._print_model_results(model_name, results, y_pred)
        
        # Дополнительная информация для MLP
        print(f"\n🧠 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        print(f"  Количество итераций обучения: {model.n_iter_}")
        print(f"  Финальное значение функции потерь: {model.loss_:.4f}")
        print(f"  Количество скрытых слоев: {len(model.hidden_layer_sizes)}")
        
        self.results[model_name] = results
        
        return results
    
    def run_all_models(self):
        """Запуск всех моделей с подробным выводом"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК ПОДРОБНОГО СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        
        # Список моделей для оценки
        models_to_evaluate = [
            ("Logistic Regression", self.evaluate_logistic_regression),
            ("SVM (Linear Kernel)", self.evaluate_svm_linear),
            ("SVM (RBF Kernel)", self.evaluate_svm_rbf),
            ("Random Forest", self.evaluate_random_forest),
            ("K-Nearest Neighbors", self.evaluate_knn),
            ("Decision Tree", self.evaluate_decision_tree),
            ("Gaussian Naive Bayes", self.evaluate_naive_bayes),
            ("Neural Network (MLP)", self.evaluate_mlp)
        ]
        
        for model_name, evaluate_func in models_to_evaluate:
            try:
                print(f"\n▶️  НАЧИНАЮ ОЦЕНКУ: {model_name}")
                evaluate_func()
                print(f"✅ {model_name} - ОЦЕНКА ЗАВЕРШЕНА")
            except Exception as e:
                print(f"❌ Ошибка при оценке {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Сравнение результатов
        self.compare_all_results()
        
        # Визуализация
        self.visualize_comparison()
        
        # Сохранение результатов
        self.save_detailed_results()
    
    def compare_all_results(self):
        """Сравнение результатов всех моделей"""
        print("\n" + "="*80)
        print("📈 СВОДНОЕ СРАВНЕНИЕ РЕЗУЛЬТАТОВ ВСЕХ МОДЕЛЕЙ")
        print("="*80)
        
        if not self.results:
            print("⚠️  Нет результатов для сравнения")
            return
        
        # Создаем таблицу сравнения
        comparison_data = []
        
        for model_name, res in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{res['accuracy']:.4f}",
                'F1-Score': f"{res['f1_score']:.4f}",
                'Precision': f"{res['precision']:.4f}",
                'Recall': f"{res['recall']:.4f}",
                'Train Time (s)': f"{res['train_time']:.2f}",
                'Predict Time (s)': f"{res['predict_time']:.4f}"
            })
        
        # Сортируем по точности
        comparison_data.sort(key=lambda x: float(x['Accuracy']), reverse=True)
        
        # Выводим таблицу
        print("\n" + "-"*110)
        print(f"{'МОДЕЛЬ':<25} {'ТОЧНОСТЬ':<12} {'F1-МЕРА':<12} {'ТОЧНОСТЬ (prec)':<15} {'ПОЛНОТА':<12} {'ОБУЧЕНИЕ (с)':<12} {'ПРЕДСКАЗАНИЕ (с)':<15}")
        print("-"*110)
        
        for i, row in enumerate(comparison_data):
            if i == 0:
                # Выделяем лучшую модель
                print(f"🏆 {row['Model']:<23} {row['Accuracy']:<12} {row['F1-Score']:<12} "
                      f"{row['Precision']:<15} {row['Recall']:<12} "
                      f"{row['Train Time (s)']:<12} {row['Predict Time (s)']:<15}")
            else:
                print(f"   {row['Model']:<25} {row['Accuracy']:<12} {row['F1-Score']:<12} "
                      f"{row['Precision']:<15} {row['Recall']:<12} "
                      f"{row['Train Time (s)']:<12} {row['Predict Time (s)']:<15}")
        
        print("-"*110)
        
        # Статистика
        best_model = comparison_data[0]['Model']
        print(f"\n📊 СТАТИСТИКА:")
        print(f"  Лучшая модель: {best_model}")
        print(f"  Количество протестированных моделей: {len(comparison_data)}")
        print(f"  Диапазон точности: {float(comparison_data[-1]['Accuracy']):.4f} - {float(comparison_data[0]['Accuracy']):.4f}")
        print(f"  Средняя точность: {np.mean([float(row['Accuracy']) for row in comparison_data]):.4f}")
    
    def visualize_comparison(self):
        """Визуализация сравнения моделей"""
        if not self.results:
            return
        
        # Создаем фигуру с несколькими графиками
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'ДЕТАЛЬНОЕ СРАВНЕНИЕ МОДЕЛЕЙ НА ДАТАСЕТЕ "{self.dataset_name}"', 
                    fontsize=16, y=1.02)
        
        # 1. Основное сравнение точности
        ax1 = plt.subplot(2, 3, 1)
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        bars = ax1.bar(model_names, accuracies, color=plt.cm.Set3(np.arange(len(model_names))/len(model_names)))
        ax1.set_title('Сравнение точности моделей', fontsize=14)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Время обучения
        ax2 = plt.subplot(2, 3, 2)
        train_times = [self.results[name]['train_time'] for name in model_names]
        
        bars = ax2.bar(model_names, train_times, color='lightcoral')
        ax2.set_title('Время обучения моделей', fontsize=14)
        ax2.set_ylabel('Время (секунды)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, time_val in zip(bars, train_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Сравнение метрик
        ax3 = plt.subplot(2, 3, 3)
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        precisions = [self.results[name]['precision'] for name in model_names]
        recalls = [self.results[name]['recall'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax3.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
        ax3.bar(x, f1_scores, width, label='F1-Score', color='lightgreen')
        ax3.bar(x + width, precisions, width, label='Precision', color='salmon')
        
        ax3.set_title('Сравнение метрик качества', fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Матрица ошибок для лучшей модели
        ax4 = plt.subplot(2, 3, 4)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_predictions = self.results[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, best_predictions)
        
        if self.classes and len(self.classes) == cm.shape[0]:
            tick_labels = self.classes
        else:
            tick_labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=tick_labels,
                   yticklabels=tick_labels,
                   ax=ax4)
        ax4.set_title(f'Матрица ошибок: {best_model_name}', fontsize=14)
        ax4.set_xlabel('Предсказанные метки')
        ax4.set_ylabel('Истинные метки')
        
        # 5. ROC-кривые для моделей с вероятностями (только для бинарной классификации)
        ax5 = plt.subplot(2, 3, 5)
        
        if len(np.unique(self.y_test)) == 2:  # Бинарная классификация
            ax5.set_title('ROC-кривые моделей', fontsize=14)
            ax5.set_xlabel('False Positive Rate')
            ax5.set_ylabel('True Positive Rate')
            ax5.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
            
            for model_name, res in self.results.items():
                if 'predictions_proba' in res:
                    fpr, tpr, _ = roc_curve(self.y_test, res['predictions_proba'][:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax5.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            ax5.legend(loc='lower right')
            ax5.grid(True, alpha=0.3)
        else:
            # Для мультиклассовой классификации - показываем сравнение времени
            ax5.set_title('Сравнение времени обучения и предсказания', fontsize=14)
            
            predict_times = [self.results[name]['predict_time'] for name in model_names]
            
            x = np.arange(len(model_names))
            ax5.bar(x - 0.2, train_times, 0.4, label='Обучение', color='lightcoral')
            ax5.bar(x + 0.2, predict_times, 0.4, label='Предсказание', color='lightblue')
            
            ax5.set_xticks(x)
            ax5.set_xticklabels(model_names, rotation=45)
            ax5.set_ylabel('Время (секунды)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Кривая обучения для MLP (если доступна)
        ax6 = plt.subplot(2, 3, 6)
        
        if 'Neural Network (MLP)' in self.results and 'loss_curve' in self.results['Neural Network (MLP)']:
            loss_curve = self.results['Neural Network (MLP)']['loss_curve']
            ax6.plot(loss_curve, label='Функция потерь')
            ax6.set_title('Кривая обучения MLP', fontsize=14)
            ax6.set_xlabel('Итерация')
            ax6.set_ylabel('Loss')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            # Альтернатива: распределение предсказаний
            ax6.set_title('Распределение точности моделей', fontsize=14)
            ax6.hist(accuracies, bins=10, edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Точность')
            ax6.set_ylabel('Количество моделей')
            ax6.axvline(np.mean(accuracies), color='red', linestyle='--', 
                       label=f'Среднее: {np.mean(accuracies):.3f}')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_detailed_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Дополнительные визуализации
        self._create_additional_visualizations()
    
    def _create_additional_visualizations(self):
        """Создание дополнительных визуализаций"""
        # График для сравнения производительности
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # График 1: Время vs Точность
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        train_times = [self.results[name]['train_time'] for name in model_names]
        
        scatter = axes[0].scatter(train_times, accuracies, s=100, alpha=0.7)
        axes[0].set_xlabel('Время обучения (сек)')
        axes[0].set_ylabel('Точность')
        axes[0].set_title('Производительность: Время vs Точность')
        axes[0].grid(True, alpha=0.3)
        
        # Добавляем подписи
        for i, name in enumerate(model_names):
            axes[0].annotate(name, (train_times[i], accuracies[i]), 
                           fontsize=8, alpha=0.8)
        
        # График 2: F1-Score по классам для лучшей модели
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        
        if best_model_name in self.detailed_reports:
            report = self.detailed_reports[best_model_name]
            classes = self.classes if self.classes else [f'Class {i}' for i in range(len(report)-3)]
            
            # Извлекаем F1-Score для каждого класса
            f1_scores = []
            valid_classes = []
            
            for cls in classes:
                if cls in report:
                    f1_scores.append(report[cls]['f1-score'])
                    valid_classes.append(cls)
            
            if f1_scores:
                axes[1].bar(valid_classes, f1_scores, color='lightgreen')
                axes[1].set_xlabel('Класс')
                axes[1].set_ylabel('F1-Score')
                axes[1].set_title(f'F1-Score по классам ({best_model_name})')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].set_ylim(0, 1.1)
                axes[1].grid(True, alpha=0.3, axis='y')
                
                for i, (cls, score) in enumerate(zip(valid_classes, f1_scores)):
                    axes[1].text(i, score + 0.02, f'{score:.2f}', 
                               ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_performance_analysis.png', dpi=150)
        plt.show()
    
    def save_detailed_results(self):
        """Сохранение подробных результатов"""
        if not self.results:
            return
        
        # Подготавливаем данные для сохранения
        output_data = {
            'dataset_info': {
                'name': self.dataset_name,
                'classes': self.classes,
                'image_size': self.input_shape,
                'grayscale': self.grayscale,
                'train_samples': self.X_train.shape[0],
                'test_samples': self.X_test.shape[0]
            },
            'models_results': {},
            'summary': {
                'best_model': None,
                'best_accuracy': 0
            }
        }
        
        # Заполняем результаты моделей
        for name, res in self.results.items():
            output_data['models_results'][name] = {
                'accuracy': float(res['accuracy']),
                'f1_score': float(res['f1_score']),
                'precision': float(res['precision']),
                'recall': float(res['recall']),
                'train_time': float(res['train_time']),
                'predict_time': float(res['predict_time'])
            }
            
            # Обновляем лучшую модель
            if res['accuracy'] > output_data['summary']['best_accuracy']:
                output_data['summary']['best_model'] = name
                output_data['summary']['best_accuracy'] = float(res['accuracy'])
        
        # Добавляем детальные отчеты
        if self.detailed_reports:
            output_data['detailed_reports'] = self.detailed_reports
        
        # Сохраняем в JSON
        output_file = f'{self.dataset_name}_detailed_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 ПОДРОБНЫЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
        print(f"  Файл: {output_file}")
        print(f"  Лучшая модель: {output_data['summary']['best_model']}")
        print(f"  Точность лучшей модели: {output_data['summary']['best_accuracy']:.4f}")
        
        # Также сохраняем в CSV для удобства
        import pandas as pd
        csv_data = []
        for name, res in output_data['models_results'].items():
            csv_data.append({
                'Model': name,
                'Accuracy': res['accuracy'],
                'F1_Score': res['f1_score'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'Train_Time_s': res['train_time'],
                'Predict_Time_s': res['predict_time']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(f'{self.dataset_name}_results_table.csv', index=False)
        print(f"  Таблица результатов: {self.dataset_name}_results_table.csv")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Подробное сравнение моделей для распознавания образов')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Путь к файлу dataset.npz')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🤖 СИСТЕМА ПОДРОБНОГО СРАВНЕНИЯ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("="*80)
    print("Версия с детальным выводом для каждой модели")
    print("="*80)
    
    # Создаем компаратор
    comparator = DetailedModelComparator(args.dataset)
    
    # Запускаем сравнение моделей
    comparator.run_all_models()
    
    print("\n" + "="*80)
    print("🎉 СРАВНЕНИЕ МОДЕЛЕЙ УСПЕШНО ЗАВЕРШЕНО!")
    print("="*80)
    print("\n📋 ИТОГИ:")
    print("  1. Каждая модель была подробно проанализирована")
    print("  2. Результаты сохранены в файлы JSON и CSV")
    print("  3. Созданы визуализации для анализа")
    print("  4. Определена лучшая модель для вашего датасета")
    print("\n✅ Готово к использованию!")


if __name__ == "__main__":
    main()