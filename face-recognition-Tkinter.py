import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import threading

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        
        # Настройка папки с известными лицами
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Base")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # Инициализация переменных
        self.known_face_encodings = []
        self.known_face_names = []
        self.camera = None
        self.is_camera_active = False
        self.current_frame = None
        self.video_processing = False
        self.video_cap = None
        
        # Загрузка известных лиц
        self.load_known_faces()
        
        # Создание GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Левая панель - видео
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        self.video_frame.grid(row=0, column=0, rowspan=4, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(self.video_frame, background="black")
        self.video_label.pack()
        
        # Правая панель - управление
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Кнопки управления
        ttk.Button(control_frame, text="Start Camera", command=self.start_camera).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Process Image", command=self.process_image).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Process Video", command=self.process_video).pack(fill=tk.X, pady=5)
        
        # Кнопка остановки видео
        self.stop_video_btn = ttk.Button(control_frame, text="Stop Video", 
                                        command=self.stop_video_processing,
                                        state='disabled')
        self.stop_video_btn.pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Статус
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.pack(pady=5)
        
        # Информация о базе лиц
        info_frame = ttk.LabelFrame(main_frame, text="Known Faces", padding="10")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.faces_listbox = tk.Listbox(info_frame, height=10)
        self.faces_listbox.pack(fill=tk.BOTH, expand=True)
        
        for face in self.known_face_names:
            self.faces_listbox.insert(tk.END, face)
            
        ttk.Button(info_frame, text="Refresh Faces", command=self.refresh_faces).pack(fill=tk.X, pady=(5, 0))
        
        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def load_known_faces(self):
        """Загрузка известных лиц из папки Base"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(self.base_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(self.base_dir, filename)
                try:
                    # Загрузка изображения
                    image = face_recognition.load_image_file(image_path)
                    
                    # Поиск лиц на изображении
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        # Используем первое найденное лицо
                        self.known_face_encodings.append(face_encodings[0])
                        # Имя файла без расширения
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        print(f"Loaded: {name}")
                    else:
                        print(f"No faces found in {filename}")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def refresh_faces(self):
        """Обновление списка известных лиц"""
        self.load_known_faces()
        self.faces_listbox.delete(0, tk.END)
        for face in self.known_face_names:
            self.faces_listbox.insert(tk.END, face)
        self.status_label.config(text=f"Loaded {len(self.known_face_names)} faces")
    
    def start_camera(self):
        """Запуск камеры в реальном времени"""
        if self.is_camera_active:
            return
        
        self.is_camera_active = True
        self.status_label.config(text="Status: Camera Active")
        
        # Запуск камеры в отдельном потоке
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def stop_camera(self):
        """Остановка камеры"""
        self.is_camera_active = False
        self.status_label.config(text="Status: Camera Stopped")
    
    def camera_loop(self):
        """Основной цикл обработки видео с камеры"""
        cap = cv2.VideoCapture(0)
        
        # Настройки для macOS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.is_camera_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ОТЗЕРКАЛИВАНИЕ КАМЕРЫ (как в зеркале)
            frame = cv2.flip(frame, 1)  # 1 - горизонтальное отзеркаливание
            
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            
            # Конвертация для отображения в Tkinter
            self.current_frame = processed_frame
            self.root.after(1, self.update_video_display)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_frame(self, frame):
        """Обработка одного кадра для распознавания лиц"""
        # Уменьшаем размер для быстрой обработки
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Конвертация цветов BGR (OpenCV) в RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Поиск лиц на кадре
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Сравнение с известными лицами
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            if len(self.known_face_encodings) > 0:
                # Вычисление расстояний
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        
        # Отображение результатов на оригинальном кадре
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Масштабирование координат обратно
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # ВЫБОР ЦВЕТА: красный для Unknown, зеленый для известных
            if name == "Unknown":
                color = (0, 0, 255)  # Красный в формате BGR
            else:
                color = (0, 255, 0)  # Зеленый в формате BGR
            
            # Рисуем рамку вокруг лица
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Рисуем подпись с именем
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
        
        return frame
    
    def update_video_display(self):
        """Обновление отображения видео в Tkinter"""
        if self.current_frame is not None:
            # Конвертация BGR в RGB
            rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # Конвертация в ImageTk
            image = Image.fromarray(rgb_image)
            image.thumbnail((800, 600))  # Изменение размера для отображения
            photo = ImageTk.PhotoImage(image=image)
            
            self.video_label.configure(image=photo)
            self.video_label.image = photo
    
    def process_image(self):
        """Обработка одиночного изображения"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.status_label.config(text=f"Processing: {os.path.basename(file_path)}")
            
            # Чтение изображения
            image = cv2.imread(file_path)
            if image is None:
                self.status_label.config(text="Error: Could not read image")
                return
            
            # Обработка
            processed_image = self.process_frame(image)
            
            # Отображение результата в главном окне
            self.current_frame = processed_image
            self.update_video_display()
            self.status_label.config(text="Image processed")
            
            # СОЗДАЕМ ОТДЕЛЬНОЕ ОКНО Tkinter для результата
            result_window = tk.Toplevel(self.root)
            result_window.title("Processed Image")
            result_window.geometry("800x600")
            
            # Конвертация для отображения в Tkinter
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_image)
            
            # Изменяем размер изображения для окна
            image_pil.thumbnail((780, 550), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=image_pil)
            
            # Отображение изображения в новом окне
            img_label = ttk.Label(result_window, image=photo)
            img_label.image = photo  # Сохраняем ссылку
            img_label.pack(padx=10, pady=10)
            
            # Кнопка закрытия окна
            ttk.Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)
            
            # Когда окно закрывается, возвращаем статус
            def on_window_close():
                self.status_label.config(text="Ready")
            
            result_window.protocol("WM_DELETE_WINDOW", lambda: [result_window.destroy(), on_window_close()])
    
    def process_video(self):
        """Обработка видеофайла"""
        if self.video_processing:
            messagebox.showwarning("Warning", "Video is already processing!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4")]
        )
        
        if not file_path:
            return
            
        # Открываем видео в главном потоке для проверки
        self.video_cap = cv2.VideoCapture(file_path)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", "Could not open video file!")
            self.video_cap = None
            return
        
        # Включаем кнопку остановки
        self.video_processing = True
        self.stop_video_btn.config(state='normal')
        self.status_label.config(text="Processing video... Click 'Stop Video' to stop")
        
        # Получаем FPS видео для плавного воспроизведения
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Значение по умолчанию
        delay = int(1000 / fps)
        
        # Запускаем обработку видео через основной цикл Tkinter
        self.process_video_frame(file_path, delay)
    
    def process_video_frame(self, file_path, delay):
        """Обработка одного кадра видео (вызывается рекурсивно)"""
        if not self.video_processing:
            self.video_cap.release()
            self.video_cap = None
            self.stop_video_btn.config(state='disabled')
            self.status_label.config(text="Video processing stopped")
            return
            
        ret, frame = self.video_cap.read()
        if not ret:
            # Видео закончилось
            self.video_processing = False
            self.video_cap.release()
            self.video_cap = None
            self.stop_video_btn.config(state='disabled')
            self.status_label.config(text="Video processing completed")
            messagebox.showinfo("Info", "Video processing completed!")
            return
        
        # Обработка кадра
        processed_frame = self.process_frame(frame)
        
        # Отображение в основном окне
        self.current_frame = processed_frame
        self.update_video_display()
        
        # Рекурсивный вызов для следующего кадра
        self.root.after(delay, lambda: self.process_video_frame(file_path, delay))
    
    def stop_video_processing(self):
        """Остановка обработки видео"""
        self.video_processing = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        
        self.stop_video_btn.config(state='disabled')
        self.status_label.config(text="Video processing stopped")

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()