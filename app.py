import cv2
import math
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
import os
import sys
from datetime import datetime

# Mengatur Appearance Mode & Theme CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PhotoStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Studio Pas Foto Professional - AI Auto Crop & Background")
        self.geometry("1100x700")
        self.minsize(900, 600)

        # Inisialisasi MediaPipe Solutions dengan aman
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=1, 
            min_detection_confidence=0.5
        )
        
        self.mp_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_segmentation.SelfieSegmentation(model_selection=0)

        # Variabel Aplikasi
        self.cap = None
        self.is_camera_open = False
        self.current_frame = None
        self.captured_image = None
        self.processed_image = None
        self.bg_color = (0, 0, 255) # Default Red (BGR)

        self.setup_ui()

    def setup_ui(self):
        # Grid Layout (1 Baris, 2 Kolom)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Panel Kiri - Preview Kamera / Hasil
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.preview_label = ctk.CTkLabel(
            self.left_frame, 
            text="Kamera Belum Aktif\nKlik 'Buka Kamera' Untuk Memulai", 
            font=("Arial", 16)
        )
        self.preview_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Panel Kanan - Kontrol Fitur
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Kontrol Kamera
        self.btn_camera = ctk.CTkButton(self.right_frame, text="Buka Kamera", command=self.toggle_camera)
        self.btn_camera.pack(padx=20, pady=10, fill="x")

        self.btn_capture = ctk.CTkButton(self.right_frame, text="Ambil Foto", command=self.capture_photo, state="disabled")
        self.btn_capture.pack(padx=20, pady=10, fill="x")

        # Pilihan Warna Background
        self.lbl_bg = ctk.CTkLabel(self.right_frame, text="Pilih Warna Background:", font=("Arial", 12, "bold"))
        self.lbl_bg.pack(padx=20, pady=(20, 5), anchor="w")

        self.bg_option = ctk.CTkOptionMenu(
            self.right_frame,
            values=["Merah", "Biru", "Putih", "Polos / Transparan"],
            command=self.change_bg_color
        )
        self.bg_option.pack(padx=20, pady=5, fill="x")

        # Tombol Aksi
        self.btn_process = ctk.CTkButton(self.right_frame, text="Proses Pas Foto (AI)", command=self.process_pas_foto, state="disabled")
        self.btn_process.pack(padx=20, pady=(30, 10), fill="x")

        self.btn_save = ctk.CTkButton(
            self.right_frame, 
            text="Simpan Hasil", 
            command=self.save_photo, 
            state="disabled", 
            fg_color="green", 
            hover_color="darkgreen"
        )
        self.btn_save.pack(padx=20, pady=10, fill="x")

    def toggle_camera(self):
        if not self.is_camera_open:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW untuk performa optimal di Windows
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0) # Fallback jika DSHOW tidak didukung
                
            if not self.cap.isOpened():
                self.preview_label.configure(text="Gagal Membuka Kamera!\nPastikan Kamera Terhubung.")
                return

            self.is_camera_open = True
            self.btn_camera.configure(text="Tutup Kamera")
            self.btn_capture.configure(state="normal")
            self.update_camera()
        else:
            self.is_camera_open = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.btn_camera.configure(text="Buka Kamera")
            self.btn_capture.configure(state="disabled")
            self.preview_label.configure(text="Kamera Dimatikan", image="")

    def update_camera(self):
        if self.is_camera_open and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                # Mirror preview
                display_frame = cv2.flip(frame, 1)
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Resizing responsive
                w = max(100, self.preview_label.winfo_width())
                h = max(100, self.preview_label.winfo_height())
                pil_image.thumbnail((w, h))

                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
                self.preview_label.configure(image=ctk_image, text="")
            
            self.after(20, self.update_camera)

    def capture_photo(self):
        if self.current_frame is not None:
            self.captured_image = self.current_frame.copy()
            self.btn_process.configure(state="normal")
            
            # Tampilkan static preview hasil tangkapan
            rgb_image = cv2.cvtColor(cv2.flip(self.captured_image, 1), cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            w = max(100, self.preview_label.winfo_width())
            h = max(100, self.preview_label.winfo_height())
            pil_image.thumbnail((w, h))
            
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
            self.preview_label.configure(image=ctk_image, text="Foto Berhasil Diambil!")
            
            # Otomatis matikan live stream kamera
            self.toggle_camera()

    def change_bg_color(self, choice):
        if choice == "Merah":
            self.bg_color = (0, 0, 255) # BGR Red
        elif choice == "Biru":
            self.bg_color = (255, 0, 0) # BGR Blue
        elif choice == "Putih":
            self.bg_color = (255, 255, 255) # BGR White
        elif choice == "Polos / Transparan":
            self.bg_color = None

    def process_pas_foto(self):
        if self.captured_image is None:
            return

        image = cv2.flip(self.captured_image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. AI Selfie Segmentation
        seg_result = self.segmentation.process(rgb_image)
        if seg_result.segmentation_mask is not None:
            condition = np.stack((seg_result.segmentation_mask,) * 3, axis=-1) > 0.5
            if self.bg_color is not None:
                bg_img = np.zeros(image.shape, dtype=np.uint8)
                bg_img[:] = self.bg_color
                output_image = np.where(condition, image, bg_img)
            else:
                output_image = image
        else:
            output_image = image.copy()

        # 2. AI Auto-Crop Pas Foto (Rasio 3:4)
        pose_result = self.pose.process(rgb_image)
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            center_x = int(nose.x * w)
            top_y = max(0, int((nose.y - 0.25) * h))
            shoulder_y = int(max(left_shoulder.y, right_shoulder.y) * h)
            bottom_y = min(h, int(shoulder_y + (shoulder_y - top_y) * 0.3))

            crop_h = bottom_y - top_y
            crop_w = int(crop_h * (3 / 4))

            left_x = max(0, center_x - crop_w // 2)
            right_x = min(w, left_x + crop_w)

            if right_x - left_x > 20 and bottom_y - top_y > 20:
                output_image = output_image[top_y:bottom_y, left_x:right_x]

        self.processed_image = output_image
        self.btn_save.configure(state="normal")

        # Tampilkan Hasil Akhir
        display_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_rgb)
        lbl_w = max(100, self.preview_label.winfo_width())
        lbl_h = max(100, self.preview_label.winfo_height())
        pil_image.thumbnail((lbl_w, lbl_h))
        
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        self.preview_label.configure(image=ctk_image, text="")

    def save_photo(self):
        if self.processed_image is not None:
            save_dir = "Hasil_Pas_Foto"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            filename = os.path.join(save_dir, f"PasFoto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(filename, self.processed_image)
            self.preview_label.configure(text=f"Foto Berhasil Disimpan!\nLokasi: {os.path.abspath(filename)}")

if __name__ == "__main__":
    app = PhotoStudioApp()
    app.mainloop()
