import cv2
import math
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
import os
from datetime import datetime

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PhotoStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Studio Pas Foto Professional - AI Auto Crop & Background")
        self.geometry("1100x700")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)
        self.mp_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_segmentation.SelfieSegmentation(model_selection=1)

        self.SIZES = {"2x3": (330, 450), "3x4": (354, 472), "4x6": (450, 668)}
        self.BG_COLORS = {"MERAH": (0, 0, 218), "BIRU": (218, 112, 0), "PUTIH": (255, 255, 255), "ASLI": None}

        self.current_size_name = "3x4"
        self.current_bg_name = "MERAH"
        self.last_cropped_pas_foto = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.setup_ui()
        self.update_video_feed()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.left_panel, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.right_panel = ctk.CTkFrame(self, width=320)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(self.right_panel, text="KONTROL FOTO", font=("Arial", 18, "bold")).pack(pady=10)
        self.preview_label = ctk.CTkLabel(self.right_panel, text="Preview Pas Foto", width=200, height=260)
        self.preview_label.pack(pady=10)

        ctk.CTkLabel(self.right_panel, text="Pilih Ukuran:", font=("Arial", 12, "bold")).pack(anchor="w", padx=20, pady=(10, 2))
        self.size_option = ctk.CTkOptionMenu(self.right_panel, values=list(self.SIZES.keys()), command=self.change_size_event)
        self.size_option.set("3x4")
        self.size_option.pack(padx=20, fill="x", pady=2)

        ctk.CTkLabel(self.right_panel, text="Pilih Background:", font=("Arial", 12, "bold")).pack(anchor="w", padx=20, pady=(10, 2))
        self.bg_option = ctk.CTkOptionMenu(self.right_panel, values=list(self.BG_COLORS.keys()), command=self.change_bg_event)
        self.bg_option.set("MERAH")
        self.bg_option.pack(padx=20, fill="x", pady=2)

        self.btn_save_single = ctk.CTkButton(self.right_panel, text="Simpan Pas Foto", fg_color="#1f538d", command=self.save_single_photo)
        self.btn_save_single.pack(padx=20, pady=(25, 5), fill="x")

        self.btn_save_sheet = ctk.CTkButton(self.right_panel, text="Buat Lembaran Cetak (A4)", fg_color="#27a770", command=self.export_a4_print_sheet)
        self.btn_save_sheet.pack(padx=20, pady=5, fill="x")

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            live_preview, pas_foto = self.process_pas_foto(frame, target_size=self.SIZES[self.current_size_name], bg_color=self.BG_COLORS[self.current_bg_name])

            img_rgb = cv2.cvtColor(live_preview, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            self.video_label.configure(image=img_tk)

            if pas_foto is not None:
                self.last_cropped_pas_foto = pas_foto.copy()
                pas_rgb = cv2.cvtColor(pas_foto, cv2.COLOR_BGR2RGB)
                pas_pil = Image.fromarray(pas_rgb)
                w, h = self.SIZES[self.current_size_name]
                display_h = 220
                display_w = int(w * (display_h / h))
                pas_tk = ctk.CTkImage(light_image=pas_pil, dark_image=pas_pil, size=(display_w, display_h))
                self.preview_label.configure(image=pas_tk, text="")

        self.after(20, self.update_video_feed)

    def rotate_image(self, image, angle, center):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def change_background(self, image, color_bgr):
        if color_bgr is None:
            return image
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb_img)
        mask = results.segmentation_mask
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = color_bgr
        mask_3d = np.stack((mask,) * 3, axis=-1)
        output_image = np.where(mask_3d > 0.6, image, bg_image)
        return output_image.astype(np.uint8)

    def process_pas_foto(self, frame, target_size, bg_color):
        h_img, w_img, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return frame, None

        landmarks = results.pose_landmarks.landmark
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

        lx, ly = int(l_shoulder.x * w_img), int(l_shoulder.y * h_img)
        rx, ry = int(r_shoulder.x * w_img), int(r_shoulder.y * h_img)
        nx, ny = int(nose.x * w_img), int(nose.y * h_img)

        dx = lx - rx
        dy = ly - ry
        angle_deg = math.degrees(math.atan2(dy, dx))
        center_point = (int((lx + rx) / 2), int((ly + ry) / 2))
        rotated_frame = self.rotate_image(frame, angle_deg, center_point)

        processed_bg = self.change_background(rotated_frame, bg_color)

        shoulder_width = math.sqrt(dx**2 + dy**2)
        target_w, target_h = target_size
        aspect_ratio = target_h / target_w

        crop_w = int(shoulder_width * 1.85)
        crop_h = int(crop_w * aspect_ratio)

        center_y = int(ny + (center_point[1] - ny) * 0.45)
        x1 = max(0, center_point[0] - crop_w // 2)
        y1 = max(0, center_y - int(crop_h * 0.35))
        x2 = min(w_img, x1 + crop_w)
        y2 = min(h_img, y1 + crop_h)

        cropped = processed_bg[y1:y2, x1:x2]
        if cropped.size > 0:
            final_pas_foto = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
            cv2.rectangle(rotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return rotated_frame, final_pas_foto

        return frame, None

    def change_size_event(self, new_size):
        self.current_size_name = new_size

    def change_bg_event(self, new_bg):
        self.current_bg_name = new_bg

    def save_single_photo(self):
        if self.last_cropped_pas_foto is not None:
            if not os.path.exists("output"):
                os.makedirs("output")
            filename = f"output/pas_foto_{self.current_size_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.last_cropped_pas_foto)

    def export_a4_print_sheet(self):
        if self.last_cropped_pas_foto is None:
            return
        if not os.path.exists("output"):
            os.makedirs("output")
        a4_w, a4_h = 2480, 3508
        canvas = np.ones((a4_h, a4_w, 3), dtype=np.uint8) * 255
        photo = self.last_cropped_pas_foto
        ph, pw, _ = photo.shape
        margin_x, margin_y = 150, 150
        gap_x, gap_y = 40, 40
        cols = (a4_w - 2 * margin_x) // (pw + gap_x)
        rows = (a4_h - 2 * margin_y) // (ph + gap_y)

        for r in range(rows):
            for c in range(cols):
                x = margin_x + c * (pw + gap_x)
                y = margin_y + r * (ph + gap_y)
                canvas[y:y+ph, x:x+pw] = photo

        sheet_filename = f"output/lembar_cetak_A4_{self.current_size_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(sheet_filename, canvas)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = PhotoStudioApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
