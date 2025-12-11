# main_system.py - ГЛАВНАЯ СИСТЕМА С ИНТЕГРАЦИЕЙ ВЫБОРА МОДЕЛИ
"""
Главная система распознавания образов с интеграцией выбора модели.

Этот модуль предоставляет интерактивный интерфейс для:
- Подготовки данных
- Обучения моделей
- Предсказания с выбором модели
- Сравнения моделей
"""

import argparse
import sys
import subprocess
from pathlib import Path
import os
import json
from datetime import datetime

def print_banner():
    """Печать баннера системы"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          СИСТЕМА РАСПОЗНАВАНИЯ ОБРАЗОВ - ГЛАВНОЕ МЕНЮ            ║
    ║          С ВЫБОРОМ МОДЕЛИ ДЛЯ ПРЕДСКАЗАНИЙ                        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_modules():
    """Проверка наличия всех необходимых модулей"""
    required_modules = [
        "data_preparer.py",
        "train_and_save.py",
        "prediction_with_choice.py"
    ]
    
    missing = []
    for module in required_modules:
        if not Path(module).exists():
            missing.append(module)
    
    return missing

def run_data_preparation():
    """Запуск подготовки данных"""
    print("\n" + "="*60)
    print("🏗️  ЗАПУСК ПОДГОТОВКИ ДАННЫХ")
    print("="*60)
    
    try:
        if Path("data_preparer.py").exists():
            print("Запуск модуля подготовки данных...")
            result = subprocess.run([sys.executable, "data_preparer.py"], 
                                  capture_output=False, text=True)
            return result.returncode == 0
        else:
            print("❌ Файл data_preparer.py не найден!")
            return False
    except Exception as e:
        print(f"❌ Ошибка при запуске подготовки данных: {e}")
        return False

def run_model_training():
    """Запуск обучения моделей"""
    print("\n" + "="*60)
    print("🎯 ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    # Ищем датасеты
    npz_files = list(Path('.').rglob('dataset.npz'))
    
    if not npz_files:
        print("❌ Файлы dataset.npz не найдены!")
        print("Сначала создайте датасет с помощью режима 1")
        return False
    
    print("\n📁 Найдены датасеты:")
    for i, file in enumerate(npz_files):
        print(f"  {i+1}. {file}")
    
    while True:
        try:
            choice = input("\n👉 Выберите датасет (номер) или введите путь: ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(npz_files):
                    dataset_path = str(npz_files[idx])
                    break
                else:
                    print(f"❌ Номер должен быть от 1 до {len(npz_files)}")
            else:
                dataset_path = choice
                if Path(dataset_path).exists():
                    break
                else:
                    print(f"❌ Файл не найден: {dataset_path}")
        except:
            print("❌ Ошибка ввода. Попробуйте снова.")
    
    try:
        print(f"\nЗапуск обучения на датасете: {dataset_path}")
        result = subprocess.run(
            [sys.executable, "train_and_save.py", "--dataset", dataset_path],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка при запуске обучения: {e}")
        return False

def run_prediction_with_model_selection():
    """Запуск предсказаний с выбором модели"""
    print("\n" + "="*60)
    print("🔍 ПРЕДСКАЗАНИЯ С ВЫБОРОМ МОДЕЛИ")
    print("="*60)
    
    # Проверяем наличие обученных моделей
    models_dir = Path("trained_models")
    if not models_dir.exists():
        print("❌ Папка с обученными моделями не найдена!")
        print("Сначала обучите модели с помощью режима 2")
        return False
    
    # Проверяем наличие файла prediction_with_choice.py
    if not Path("prediction_with_choice.py").exists():
        print("❌ Файл prediction_with_choice.py не найден!")
        print("Убедитесь, что файл находится в той же папке")
        return False
    
    # Показываем меню режимов предсказания
    print("\n📋 ВЫБЕРИТЕ РЕЖИМ ПРЕДСКАЗАНИЯ:")
    print("1. 🔍 Предсказание для одного изображения")
    print("2. 📁 Предсказание для папки с изображениями")
    print("3. 📊 Показать список доступных моделей")
    print("4. 🔄 Сравнение моделей")
    print("5. 📋 Информация о загруженной модели")
    print("6. ↩️  Назад в главное меню")
    
    while True:
        try:
            choice = input("\n👉 Ваш выбор (1-6): ").strip()
            
            if choice == '1':
                # Режим одного изображения
                image_path = input("Введите путь к изображению: ").strip()
                if not Path(image_path).exists():
                    print(f"❌ Изображение не найдено: {image_path}")
                    continue
                
                cmd = [sys.executable, "prediction_with_choice.py", "--select-model", "--image", image_path]
                subprocess.run(cmd, capture_output=False, text=True)
                break
            
            elif choice == '2':
                # Режим папки с изображениями
                folder_path = input("Введите путь к папке: ").strip()
                if not Path(folder_path).exists():
                    print(f"❌ Папка не найдена: {folder_path}")
                    continue
                
                max_images = input("Максимальное количество изображений (по умолчанию 10): ").strip()
                
                cmd = [sys.executable, "prediction_with_choice.py", "--select-model", "--folder", folder_path]
                if max_images:
                    cmd.extend(["--max-images", max_images])
                
                subprocess.run(cmd, capture_output=False, text=True)
                break
            
            elif choice == '3':
                # Показать список моделей
                cmd = [sys.executable, "prediction_with_choice.py", "--list-models"]
                subprocess.run(cmd, capture_output=False, text=True)
                break
            
            elif choice == '4':
                # Сравнение моделей
                image_path = input("Введите путь к тестовому изображению: ").strip()
                if not Path(image_path).exists():
                    print(f"❌ Изображение не найдено: {image_path}")
                    continue
                
                cmd = [sys.executable, "prediction_with_choice.py", "--select-model", "--compare", "--image", image_path]
                subprocess.run(cmd, capture_output=False, text=True)
                break
            
            elif choice == '5':
                # Информация о модели
                cmd = [sys.executable, "prediction_with_choice.py", "--info"]
                subprocess.run(cmd, capture_output=False, text=True)
                break
            
            elif choice == '6':
                # Назад
                return True
            
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Операция прервана пользователем.")
            return True
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    return True

def run_advanced_model_comparison():
    """Запуск продвинутого сравнения моделей"""
    print("\n" + "="*60)
    print("📊 ПРОДВИНУТОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    # Проверяем наличие обученных моделей
    models_dir = Path("trained_models")
    if not models_dir.exists():
        print("❌ Папка с обученными моделями не найдена!")
        print("Сначала обучите модели с помощью режима 2")
        return False
    
    # Показываем доступные отчеты
    report_files = list(models_dir.glob("*.json")) + list(models_dir.glob("*.csv"))
    
    if report_files:
        print("\n📊 ДОСТУПНЫЕ ОТЧЕТЫ:")
        for i, file in enumerate(report_files):
            print(f"  {i+1}. {file.name}")
        
        # Также показываем графики
        plot_files = list(models_dir.glob("*.png"))
        if plot_files:
            print("\n📈 ГРАФИКИ СРАВНЕНИЯ:")
            for i, file in enumerate(plot_files):
                print(f"  {i+1}. {file.name}")
        
        # Предлагаем открыть отчет
        choice = input("\nОткрыть сводный отчет? (y/n): ").lower()
        if choice == 'y':
            try:
                # Создаем простой HTML отчет
                self._create_html_report(models_dir)
                
                # Пробуем открыть в браузере
                html_path = models_dir / "training_report.html"
                if os.name == 'nt':  # Windows
                    os.startfile(html_path)
                elif os.name == 'posix':  # macOS, Linux
                    subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(html_path)])
                
                print(f"✅ Отчет открыт в браузере: {html_path}")
            except:
                print("⚠️  Не удалось открыть отчет в браузере")
                print(f"Открыть файл вручную: {models_dir / 'training_report.html'}")
    
    else:
        print("ℹ️  Отчеты не найдены. Сначала обучите модели.")
    
    return True

def _create_html_report(models_dir):
    """Создание HTML отчета о моделях"""
    import json
    
    report_path = models_dir / "training_report.json"
    
    if not report_path.exists():
        # Создаем базовый отчет
        html_content = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <title>Отчет о моделях распознавания</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #4CAF50; color: white; padding: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .model { background: #f9f9f9; margin: 10px 0; padding: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет о моделях распознавания образов</h1>
                <p>Дата генерации: """ + str(Path.cwd()) + """</p>
            </div>
            <div class="section">
                <h2>Доступные модели</h2>
                <p>Для просмотра детальных отчетов запустите сравнение моделей.</p>
            </div>
        </body>
        </html>
        """
    else:
        # Загружаем существующий отчет
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # Создаем HTML на основе отчета
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <title>Отчет: {report_data.get('dataset_name', 'Неизвестный датасет')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .model {{ background: #f9f9f9; margin: 10px 0; padding: 10px; }}
                .best-model {{ background: #e8f5e9; border-left: 5px solid #4CAF50; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Отчет об обучении моделей</h1>
                <p>Датасет: {report_data.get('dataset_name', 'Неизвестный')}</p>
                <p>Дата обучения: {report_data.get('training_date', 'Неизвестно')}</p>
            </div>
        """
        
        # Лучшая модель
        if 'best_model' in report_data:
            best = report_data['best_model']
            html_content += f"""
            <div class="section">
                <h2>🏆 Лучшая модель</h2>
                <div class="model best-model">
                    <h3>{best.get('name', 'Неизвестно')}</h3>
                    <p><strong>Точность:</strong> {best.get('accuracy', 0):.4f}</p>
                    <p><strong>F1-мера:</strong> {best.get('f1_score', 0):.4f}</p>
                </div>
            </div>
            """
        
        # Сравнение моделей
        if 'models' in report_data:
            html_content += """
            <div class="section">
                <h2>📈 Сравнение всех моделей</h2>
                <table>
                    <tr>
                        <th>Модель</th>
                        <th>Точность</th>
                        <th>F1-мера</th>
                        <th>Время обучения (с)</th>
                    </tr>
            """
            
            for model_name, model_data in report_data['models'].items():
                html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{model_data.get('accuracy', 0):.4f}</td>
                        <td>{model_data.get('f1_score', 0):.4f}</td>
                        <td>{model_data.get('train_time', 0):.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
    
    # Сохраняем HTML файл
    html_path = models_dir / "training_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

def run_quick_test():
    """Быстрый тест системы"""
    print("\n" + "="*60)
    print("⚡ БЫСТРЫЙ ТЕСТ СИСТЕМЫ")
    print("="*60)
    
    print("\n🔍 Проверка модулей...")
    missing = check_modules()
    
    if missing:
        print("❌ Отсутствуют модули:")
        for module in missing:
            print(f"  - {module}")
    else:
        print("✅ Все основные модули найдены")
    
    # Проверяем датасеты
    npz_files = list(Path('.').rglob('dataset.npz'))
    if npz_files:
        print(f"✅ Найдено датасетов: {len(npz_files)}")
        for file in npz_files[:3]:  # Показываем первые 3
            print(f"  - {file}")
        if len(npz_files) > 3:
            print(f"  ... и еще {len(npz_files) - 3}")
    else:
        print("❌ Датасеты не найдены")
    
    # Проверяем модели
    models_dir = Path("trained_models")
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.joblib")) + list(models_dir.rglob("*.pkl"))
        if model_files:
            print(f"✅ Найдено моделей: {len(model_files)}")
        else:
            print("ℹ️  Папка с моделями есть, но сами модели не найдены")
    else:
        print("ℹ️  Папка с моделями не найдена")
    
    print("\n📋 РЕКОМЕНДАЦИИ:")
    if missing:
        print("  1. Убедитесь, что все модули находятся в одной папке")
    
    if not npz_files:
        print("  2. Создайте датасет с помощью режима 1")
    
    if not models_dir.exists() or not list(models_dir.rglob("*.joblib")):
        print("  3. Обучите модели с помощью режима 2")
    
    return True

def main_interactive():
    """Интерактивный режим работы"""
    while True:
        print_banner()
        
        # Показываем статус системы
        print("\n📊 СТАТУС СИСТЕМЫ:")
        
        # Проверяем датасеты
        npz_files = list(Path('.').rglob('dataset.npz'))
        if npz_files:
            print(f"  ✅ Датасеты: {len(npz_files)} найдено")
        else:
            print(f"  ⚠️  Датасеты: не найдены")
        
        # Проверяем модели
        models_dir = Path("trained_models")
        if models_dir.exists():
            model_files = list(models_dir.rglob("*.joblib")) + list(models_dir.rglob("*.pkl"))
            if model_files:
                print(f"  ✅ Модели: {len(model_files)} обучено")
            else:
                print(f"  ⚠️  Модели: папка есть, но модели не обучены")
        else:
            print(f"  ⚠️  Модели: не обучены")
        
        print("\n📋 ГЛАВНОЕ МЕНЮ:")
        print("1. 🏗️  Подготовка данных (создание нового датасета)")
        print("2. 🎯 Обучение и сохранение моделей")
        print("3. 🔍 Предсказание с выбором модели (НОВОЕ!)")
        print("4. 📊 Продвинутое сравнение моделей")
        print("5. ⚡ Быстрый тест системы")
        print("6. ❌ Выход из программы")
        
        try:
            choice = input("\n👉 Ваш выбор (1-6): ").strip()
            
            if choice == '1':
                run_data_preparation()
                
            elif choice == '2':
                run_model_training()
                
            elif choice == '3':
                run_prediction_with_model_selection()
                
            elif choice == '4':
                run_advanced_model_comparison()
                
            elif choice == '5':
                run_quick_test()
                
            elif choice == '6':
                print("\n👋 Давай, бб, связь!")
                break
                
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
            
            if choice != '6':
                input("\nНажмите Enter чтобы вернуться в главное меню...")
                print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Программа завершена пользователем.")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
            input("\nНажмите Enter чтобы продолжить...")

def main_cli():
    """Режим командной строки"""
    parser = argparse.ArgumentParser(
        description='Главная система распознавания образов с выбором модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python main_system.py                    # Интерактивный режим
  python main_system.py --prepare          # Только подготовка данных
  python main_system.py --train dataset.npz # Обучение моделей
  python main_system.py --predict          # Предсказание с выбором модели
  python main_system.py --test             # Быстрый тест системы
        '''
    )
    
    parser.add_argument('--prepare', action='store_true',
                       help='Запустить только подготовку данных')
    
    parser.add_argument('--train', type=str,
                       help='Обучение моделей на указанном датасете')
    
    parser.add_argument('--predict', action='store_true',
                       help='Запустить предсказание с выбором модели')
    
    parser.add_argument('--compare', action='store_true',
                       help='Запустить сравнение моделей')
    
    parser.add_argument('--test', action='store_true',
                       help='Запустить быстрый тест системы')
    
    args = parser.parse_args()
    
    if args.prepare:
        run_data_preparation()
    elif args.train:
        if Path(args.train).exists():
            subprocess.run([sys.executable, "train_and_save.py", "--dataset", args.train])
        else:
            print(f"❌ Файл не найден: {args.train}")
    elif args.predict:
        run_prediction_with_model_selection()
    elif args.compare:
        run_advanced_model_comparison()
    elif args.test:
        run_quick_test()
    else:
        # Если аргументов нет, запускаем интерактивный режим
        main_interactive()

if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        print("\n\n👋 Программа завершена пользователем.")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("Попробуйте запустить с аргументом --test для диагностики")