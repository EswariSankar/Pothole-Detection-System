import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
import os
import threading
import time
import math
from PIL import Image, ImageTk

# ── Model file ─────────────────────────────────────────────────
model_file = "pothole_model.pkl"

# ── Design Tokens (matching screenshot) ──────────────────────────
BG_MAIN      = "#0D1117"
BG_CARD      = "#161B22"
BG_PANEL     = "#1C2333"
BG_INPUT     = "#21262D"
BG_HOVER     = "#2D333B"
ACCENT       = "#F78166"      # orange-red
ACCENT_ALT   = "#FF9800"      # orange for train button
SUCCESS      = "#3FB950"
DANGER       = "#F85149"
INFO         = "#58A6FF"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#8B949E"
TEXT_DIM     = "#484F58"
BORDER       = "#30363D"
BORDER_MED   = "#21262D"
LIVE_GREEN   = "#3FB950"

# ══════════════════════════════════════════════════════════════
#  CORE LOGIC
# ══════════════════════════════════════════════════════════════
def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Unable to read image: {img_path}")

        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── 1. HOG features ──────────────────────────────────────
        hog = cv2.HOGDescriptor(
            _winSize=(128, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        hog_feat = hog.compute(gray).flatten()
        hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-6)

        # ── 2. Texture — Laplacian variance (surface roughness) ──
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # ── 3. Edge density — potholes have strong irregular edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # ── 4. Intensity stats — potholes are often darker patches
        mean_intensity = np.mean(gray) / 255.0
        std_intensity  = np.std(gray)  / 255.0

        # ── 5. Dark region ratio — pothole shadows
        dark_ratio = np.sum(gray < 80) / gray.size

        # ── 6. Local variance map — uneven surfaces
        # Divide image into 4×4 grid, compute variance per cell
        cell_vars = []
        h, w = gray.shape
        for i in range(4):
            for j in range(4):
                cell = gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                cell_vars.append(np.var(cell) / 10000.0)
        cell_vars = np.array(cell_vars)

        # ── Combine all signals ──────────────────────────────────
        extra = np.array([
            lap_var    / 10000.0,
            edge_density,
            mean_intensity,
            std_intensity,
            dark_ratio,
        ])

        return np.concatenate([hog_feat, extra, cell_vars])

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def set_status(msg, color=None):
    status_var.set(f"  {msg}")
    status_label.config(fg=color or TEXT_MUTED)
    root.update_idletasks()


def train_model():
    train_path = filedialog.askdirectory(title="Select Training Dataset Folder")
    if not train_path:
        return
    # Update the path display
    path_var.set(os.path.basename(train_path) or train_path)

    def run():
        train_btn.config(state="disabled", text="  ⏳  Training…")
        progress_var.set(0)
        set_status("Scanning dataset folders…", ACCENT_ALT)
        update_stat("model_stat_val", "Training…", ACCENT_ALT)
        update_stat("model_stat_sub", "Please wait", TEXT_MUTED)

        folders = sorted([
            f for f in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, f))
        ])

        if len(folders) < 2:
            messagebox.showerror("Error", "Dataset must have at least 2 class subfolders.")
            train_btn.config(state="normal", text="  ⚙  Train Model")
            set_status("Training cancelled — need at least 2 class folders.", DANGER)
            return

        pothole_folders   = [f for f in folders if "pothole" in f.lower()
                             and not any(x in f.lower() for x in ["no","plain","clear","normal","good"])]
        nopothole_folders = [f for f in folders if f not in pothole_folders]

        label_map = {}
        for f in nopothole_folders: label_map[f] = 0
        for f in pothole_folders:   label_map[f] = 1

        X_train, y_train = [], []
        total_files = sum(len(os.listdir(os.path.join(train_path, f))) for f in folders)
        processed = 0

        for class_folder in folders:
            class_path  = os.path.join(train_path, class_folder)
            class_label = label_map[class_folder]
            set_status(f"Processing: {class_folder}…", ACCENT_ALT)
            for img_file in os.listdir(class_path):
                features = extract_features(os.path.join(class_path, img_file))
                if features is not None:
                    X_train.append(features)
                    y_train.append(class_label)
                processed += 1
                pct = int((processed / max(total_files, 1)) * 100)
                progress_var.set(pct)
                prog_pct_var.set(f"{pct}%")
                set_status(f"Extracting features… {pct}%", ACCENT_ALT)
                update_stat("img_stat_val", f"{processed:,}", TEXT_PRIMARY)
                update_stat("img_stat_sub", "Total training images", TEXT_MUTED)

        if not X_train:
            messagebox.showerror("Error", "No valid images found in dataset.")
            train_btn.config(state="normal", text="  ⚙  Train Model")
            set_status("No valid images found.", DANGER)
            return

        set_status("Scaling features…", ACCENT_ALT)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        set_status("Training SVM — please wait…", ACCENT_ALT)
        model = svm.SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
        model.fit(X_scaled, y_train)

        with open(model_file, "wb") as f:
            pickle.dump((model, scaler, label_map), f)

        progress_var.set(100)
        prog_pct_var.set("100%")
        train_btn.config(state="normal", text="  ✔  Retrain Model")
        update_stat("model_stat_val", "Ready", SUCCESS)
        update_stat("model_stat_sub", "Model is trained and active", TEXT_MUTED)
        update_stat("img_stat_val", f"{len(X_train):,}", TEXT_PRIMARY)
        update_stat("img_stat_sub", "Total training images", TEXT_MUTED)
        update_stat("class_stat_val", str(len(folders)), TEXT_PRIMARY)
        update_stat("class_stat_sub", ", ".join(folders[:3]), TEXT_MUTED)
        train_status_var.set(f"✔  Model trained on {len(X_train):,} images  ·  Classes: {', '.join(folders)}")
        train_status_lbl.config(fg=SUCCESS)
        set_status(f"Model trained on {len(X_train)} images  ·  Classes: {', '.join(folders)}", SUCCESS)
        messagebox.showinfo("Training Complete",
            f"Model trained on {len(X_train)} images.\n\nLabel assignments:\n" +
            "\n".join(f"  {v} → {k}" for k, v in sorted(label_map.items(), key=lambda x: x[1])))

    threading.Thread(target=run, daemon=True).start()

_preview_ref = None   # keep a reference so GC doesn't destroy it

def show_preview(path):
    global _preview_ref
    try:
        img = Image.open(path)
        # Fit inside the preview box (approx 380 × 170 px)
        img.thumbnail((380, 170), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        _preview_ref = photo          # prevent garbage-collection
        # Clear placeholder widgets and show the image
        for w in preview_inner.winfo_children():
            w.destroy()
        tk.Label(preview_inner, image=photo, bg=BG_INPUT).place(
            relx=0.5, rely=0.5, anchor="center")
    except Exception as e:
        print(f"Preview error: {e}")


def predict_potholes():
    img_path = filedialog.askopenfilename(
        title="Select Road Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not img_path:
        return
    show_preview(img_path)
    if not os.path.exists(model_file):
        messagebox.showerror("Error", "No trained model found.\nPlease train the model first.")
        return

    set_status("Loading model…", INFO)
    with open(model_file, "rb") as f:
        loaded = pickle.load(f)

    if len(loaded) == 3:
        svm_model, scaler, label_map = loaded
        if isinstance(label_map, list):
            label_map = {name: idx for idx, name in enumerate(label_map)}
    else:
        svm_model, scaler = loaded
        label_map = {"NoPothole": 0, "Pothole": 1}

    set_status("Analysing image…", INFO)
    features = extract_features(img_path)
    if features is None:
        messagebox.showerror("Error", "Could not process selected image.")
        set_status("Image processing failed.", DANGER)
        return

    prediction = int(svm_model.predict(scaler.transform([features]))[0])
    is_pothole = (prediction == 1)
    filename   = os.path.basename(img_path)

    # Update scanned count
    current = scan_count_var.get()
    scan_count_var.set(current + 1)
    update_stat("scan_stat_val", f"{current+1:,}", TEXT_PRIMARY)
    update_stat("scan_stat_sub", "Total images analyzed", TEXT_MUTED)

    # Update analyze section image name
    analyze_img_var.set(f"📄  {filename}")

    if is_pothole:
        update_result(
            icon="⚠", icon_fg=DANGER,
            title="POTHOLE DETECTED",
            title_fg=DANGER,
            status_text="POTHOLE", status_bg=DANGER,
            img_name=filename,
            confidence=92
        )
        set_status("⚠  Pothole detected in selected image.", DANGER)
    else:
        update_result(
            icon="✔", icon_fg=SUCCESS,
            title="ROAD CLEAR",
            title_fg=SUCCESS,
            status_text="CLEAR", status_bg=SUCCESS,
            img_name=filename,
            confidence=88
        )
        set_status("✔  No pothole detected — road surface looks clear.", SUCCESS)


def update_result(icon, icon_fg, title, title_fg, status_text, status_bg, img_name, confidence):
    result_icon_lbl.config(text=icon, fg=icon_fg)
    result_title_lbl.config(text=title, fg=title_fg)
    result_img_name_var.set(img_name)
    result_status_var.set(status_text)
    result_status_lbl.config(bg=status_bg)
    





def update_stat(key, val, color):
    stat_labels[key].config(text=val, fg=color)


# ══════════════════════════════════════════════════════════════
#  ROOT WINDOW
# ══════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("Pothole Detection System")
root.geometry("1200x760")
root.minsize(1000, 660)
root.configure(bg=BG_MAIN)

scan_count_var = tk.IntVar(value=0)
stat_labels = {}

style = ttk.Style()
style.theme_use("default")
style.configure("Orange.Horizontal.TProgressbar",
                troughcolor=BG_INPUT, background=ACCENT_ALT,
                thickness=6, borderwidth=0)

# ══════════════════════════════════════════════════════════════
#  TOP HEADER BAR
# ══════════════════════════════════════════════════════════════
header = tk.Frame(root, bg=BG_CARD, height=56)
header.pack(fill="x", side="top")
header.pack_propagate(False)
tk.Frame(header, bg=BORDER, height=1).pack(fill="x", side="bottom")

h_inner = tk.Frame(header, bg=BG_CARD)
h_inner.pack(fill="both", expand=True, padx=24)

# Left: title
title_lbl = tk.Label(h_inner, text="POTHOLE DETECTION SYSTEM",
                      font=("Segoe UI", 13, "bold"),
                      fg=TEXT_PRIMARY, bg=BG_CARD)
title_lbl.pack(side="left", pady=14)

# Right: date/time + live badge
right_h = tk.Frame(h_inner, bg=BG_CARD)
right_h.pack(side="right", pady=0)

def update_clock():
    import datetime
    now = datetime.datetime.now()
    date_var.set(now.strftime("JUN %d, %Y"))
    time_var.set(now.strftime("%I:%M:%S %p"))
    root.after(1000, update_clock)

date_var = tk.StringVar()
time_var = tk.StringVar()
update_clock()

dt_frame = tk.Frame(right_h, bg=BG_CARD)
dt_frame.pack(side="left", padx=(0, 16), pady=10)
tk.Label(dt_frame, textvariable=date_var, font=("Consolas", 8),
         fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="e")
tk.Label(dt_frame, textvariable=time_var, font=("Consolas", 9, "bold"),
         fg=INFO, bg=BG_CARD).pack(anchor="e")

# Live badge
live_badge = tk.Frame(right_h, bg=BG_CARD)
live_badge.pack(side="left", pady=16)
tk.Label(live_badge, text="●", font=("Segoe UI", 8),
         fg=LIVE_GREEN, bg=BG_CARD).pack(side="left")
tk.Label(live_badge, text=" SYSTEM LIVE", font=("Consolas", 8, "bold"),
         fg=LIVE_GREEN, bg=BG_CARD).pack(side="left")

# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT AREA
# ══════════════════════════════════════════════════════════════
main_frame = tk.Frame(root, bg=BG_MAIN)
main_frame.pack(fill="both", expand=True, padx=20, pady=16)

# ══════════════════════════════════════════════════════════════
#  TOP STAT CARDS ROW (4 cards)
# ══════════════════════════════════════════════════════════════
stats_row = tk.Frame(main_frame, bg=BG_MAIN)
stats_row.pack(fill="x", pady=(0, 14))

def stat_card(parent, icon, icon_bg, eyebrow, init_val, init_sub, val_key, sub_key, init_color=None):
    card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                    highlightbackground=BORDER)
    card.pack(side="left", fill="both", expand=True, padx=(0, 10))

    inner = tk.Frame(card, bg=BG_CARD)
    inner.pack(fill="both", expand=True, padx=16, pady=14)

    # Icon + eyebrow row
    top = tk.Frame(inner, bg=BG_CARD)
    top.pack(fill="x", anchor="w")

    icon_box = tk.Frame(top, bg=icon_bg, width=28, height=28)
    icon_box.pack(side="left")
    icon_box.pack_propagate(False)
    tk.Label(icon_box, text=icon, font=("Segoe UI", 11),
             fg="white", bg=icon_bg).place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(top, text=eyebrow, font=("Segoe UI", 7, "bold"),
             fg=TEXT_MUTED, bg=BG_CARD).pack(side="left", padx=(8,0))

    val_lbl = tk.Label(inner, text=init_val,
                        font=("Segoe UI", 20, "bold"),
                        fg=init_color or TEXT_PRIMARY, bg=BG_CARD)
    val_lbl.pack(anchor="w", pady=(6, 0))
    stat_labels[val_key] = val_lbl

    sub_lbl = tk.Label(inner, text=init_sub,
                        font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_CARD)
    sub_lbl.pack(anchor="w")
    stat_labels[sub_key] = sub_lbl

stat_card(stats_row, "🛡", "#1A3A2A", "MODEL STATUS",     "Not Trained",  "Train to activate",       "model_stat_val", "model_stat_sub", DANGER)
stat_card(stats_row, "🖼", "#1A2A3A", "IMAGES PROCESSED", "0",            "Total training images",   "img_stat_val",   "img_stat_sub")
stat_card(stats_row, "📂", "#2A1A1A", "CLASSES DETECTED", "0",            "Pothole, NoPothole",      "class_stat_val", "class_stat_sub")
stat_card(stats_row, "📄", "#1A2A2A", "IMAGES SCANNED",   "0",            "Total images analyzed",   "scan_stat_val",  "scan_stat_sub")

# ── Remove last padding on the rightmost card
for w in stats_row.winfo_children():
    pass  # pack already done above; last card still has padx=(0,10) which is fine

# ══════════════════════════════════════════════════════════════
#  THREE-COLUMN MAIN CARDS
# ══════════════════════════════════════════════════════════════
cols_frame = tk.Frame(main_frame, bg=BG_MAIN)
cols_frame.pack(fill="both", expand=True)
cols_frame.grid_columnconfigure(0, weight=1)
cols_frame.grid_columnconfigure(1, weight=1)
cols_frame.grid_columnconfigure(2, weight=1)
cols_frame.grid_rowconfigure(0, weight=1)

def make_card_frame(parent, col):
    outer = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                     highlightbackground=BORDER)
    outer.grid(row=0, column=col, sticky="nsew",
               padx=(0 if col == 0 else 8, 0))
    return outer

# ── CARD 1: TRAIN AI MODEL ─────────────────────────────────────
c1 = make_card_frame(cols_frame, 0)
c1_inner = tk.Frame(c1, bg=BG_CARD)
c1_inner.pack(fill="both", expand=True, padx=20, pady=18)

# Card header
c1h = tk.Frame(c1_inner, bg=BG_CARD)
c1h.pack(fill="x")
icon_c1 = tk.Label(c1h, text="⚙️", font=("Segoe UI", 13),
                   fg=ACCENT_ALT, bg=BG_CARD)
icon_c1.pack(side="left")
tk.Label(c1h, text="  TRAIN  MODEL", font=("Segoe UI", 10, "bold"),
         fg=TEXT_PRIMARY, bg=BG_CARD).pack(side="left")

tk.Frame(c1_inner, bg=BORDER, height=1).pack(fill="x", pady=(12,14))

# Dataset folder input
tk.Label(c1_inner, text="Select Dataset Folder",
         font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0,4))

path_frame = tk.Frame(c1_inner, bg=BG_INPUT, highlightthickness=1,
                      highlightbackground=BORDER)
path_frame.pack(fill="x")
path_var = tk.StringVar(value="No folder selected")
path_lbl = tk.Label(path_frame, textvariable=path_var,
                    font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_INPUT,
                    anchor="w")
path_lbl.pack(side="left", fill="x", expand=True, padx=10, pady=8)
tk.Label(path_frame, text="📁", font=("Segoe UI", 10),
         fg=TEXT_MUTED, bg=BG_INPUT).pack(side="right", padx=8)

# Train button
train_btn = tk.Button(
    c1_inner, text="  ⚙  Train Model",
    font=("Segoe UI", 10, "bold"),
    fg="white", bg=ACCENT_ALT,
    activebackground="#E68900", activeforeground="white",
    relief="flat", cursor="hand2", pady=10,
    command=train_model
)
train_btn.pack(fill="x", pady=(14, 0))

# Progress
prog_header = tk.Frame(c1_inner, bg=BG_CARD)
prog_header.pack(fill="x", pady=(14, 4))
tk.Label(prog_header, text="Training Progress",
         font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side="left")
prog_pct_var = tk.StringVar(value="0%")
tk.Label(prog_header, textvariable=prog_pct_var,
         font=("Segoe UI", 8, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD).pack(side="right")

progress_var = tk.IntVar(value=0)
progress_bar = ttk.Progressbar(
    c1_inner, style="Orange.Horizontal.TProgressbar",
    orient="horizontal", mode="determinate",
    variable=progress_var
)
progress_bar.pack(fill="x")

train_status_var = tk.StringVar(value="")
train_status_lbl = tk.Label(c1_inner, textvariable=train_status_var,
                             font=("Segoe UI", 7), fg=SUCCESS, bg=BG_CARD,
                             wraplength=280, justify="left")
train_status_lbl.pack(anchor="w", pady=(8,0))

# ── CARD 2: ANALYZE ROAD IMAGE ─────────────────────────────────
c2 = make_card_frame(cols_frame, 1)
c2_inner = tk.Frame(c2, bg=BG_CARD)
c2_inner.pack(fill="both", expand=True, padx=20, pady=18)

c2h = tk.Frame(c2_inner, bg=BG_CARD)
c2h.pack(fill="x")
tk.Label(c2h, text="🔍", font=("Segoe UI", 13),
         fg=INFO, bg=BG_CARD).pack(side="left")
tk.Label(c2h, text="  ANALYZE ROAD IMAGE", font=("Segoe UI", 10, "bold"),
         fg=TEXT_PRIMARY, bg=BG_CARD).pack(side="left")

tk.Frame(c2_inner, bg=BORDER, height=1).pack(fill="x", pady=(12, 14))

tk.Label(c2_inner, text="Upload Road Image",
         font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0,6))

# Image preview box
preview_frame = tk.Frame(c2_inner, bg=BG_INPUT, highlightthickness=1,
                          highlightbackground=BORDER)
preview_frame.pack(fill="x", expand=False)
preview_inner = tk.Frame(preview_frame, bg=BG_INPUT, height=170)
preview_inner.pack(fill="x")
preview_inner.pack_propagate(False)

# Placeholder content in preview
ph_frame = tk.Frame(preview_inner, bg=BG_INPUT)
ph_frame.place(relx=0.5, rely=0.5, anchor="center")
tk.Label(ph_frame, text="🛣", font=("Segoe UI", 32),
         fg=TEXT_DIM, bg=BG_INPUT).pack()
tk.Label(ph_frame, text="Road image preview",
         font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_INPUT).pack(pady=(4,0))

analyze_img_var = tk.StringVar(value="")
analyze_img_lbl = tk.Label(c2_inner, textvariable=analyze_img_var,
                            font=("Consolas", 7), fg=TEXT_MUTED, bg=BG_CARD)
analyze_img_lbl.pack(anchor="w", pady=(6,0))

# Choose image button
predict_btn = tk.Button(
    c2_inner, text="  ↑  Choose Image",
    font=("Segoe UI", 10, "bold"),
    fg=TEXT_PRIMARY, bg=BG_INPUT,
    activebackground=BG_HOVER, activeforeground=TEXT_PRIMARY,
    relief="flat", cursor="hand2", pady=10,
    highlightthickness=1, highlightbackground=BORDER,
    command=predict_potholes
)
predict_btn.pack(fill="x", pady=(10, 0))

tk.Label(c2_inner,
         text="ℹ  Train the model before analyzing images.",
         font=("Segoe UI", 7), fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", pady=(8,0))

# ── CARD 3: DETECTION RESULT ───────────────────────────────────
c3 = make_card_frame(cols_frame, 2)
c3_inner = tk.Frame(c3, bg=BG_CARD)
c3_inner.pack(fill="both", expand=True, padx=20, pady=18)

c3h = tk.Frame(c3_inner, bg=BG_CARD)
c3h.pack(fill="x")
result_icon_lbl = tk.Label(c3h, text="🔎", font=("Segoe UI", 13),
                            fg=TEXT_MUTED, bg=BG_CARD)
result_icon_lbl.pack(side="left")
tk.Label(c3h, text="  DETECTION RESULT", font=("Segoe UI", 10, "bold"),
         fg=TEXT_PRIMARY, bg=BG_CARD).pack(side="left")

tk.Frame(c3_inner, bg=BORDER, height=1).pack(fill="x", pady=(12, 14))

result_title_lbl = tk.Label(c3_inner, text="Awaiting Analysis",
                             font=("Segoe UI", 14, "bold"),
                             fg=TEXT_MUTED, bg=BG_CARD)
result_title_lbl.pack(anchor="center", pady=(4,0))

# Result details
details_frame = tk.Frame(c3_inner, bg=BG_CARD)
details_frame.pack(fill="x", pady=(10, 0))

# Image name row
img_row = tk.Frame(details_frame, bg=BG_INPUT)
img_row.pack(fill="x", pady=2)
img_inner = tk.Frame(img_row, bg=BG_INPUT)
img_inner.pack(fill="x", padx=12, pady=8)
tk.Label(img_inner, text="Image Name",
         font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_INPUT).pack(side="left")
result_img_name_var = tk.StringVar(value="—")
tk.Label(img_inner, textvariable=result_img_name_var,
         font=("Consolas", 7, "bold"), fg=TEXT_PRIMARY, bg=BG_INPUT).pack(side="right")

# Status row
status_row = tk.Frame(details_frame, bg=BG_INPUT)
status_row.pack(fill="x", pady=2)
status_inner = tk.Frame(status_row, bg=BG_INPUT)
status_inner.pack(fill="x", padx=12, pady=8)
tk.Label(status_inner, text="Detection Status",
         font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_INPUT).pack(side="left")
result_status_var = tk.StringVar(value="PENDING")
result_status_lbl = tk.Label(status_inner, textvariable=result_status_var,
                              font=("Consolas", 7, "bold"),
                              fg="white", bg=TEXT_DIM, padx=6, pady=2)
result_status_lbl.pack(side="right")

# ══════════════════════════════════════════════════════════════
#  STATUS BAR
# ══════════════════════════════════════════════════════════════
tk.Frame(root, bg=BORDER, height=1).pack(fill="x", side="bottom")
status_var = tk.StringVar(value="  Ready  ·  Select a dataset folder to begin training.")
status_label = tk.Label(
    root, textvariable=status_var,
    font=("Consolas", 8), fg=TEXT_MUTED, bg=BG_CARD,
    anchor="w", padx=8, pady=6
)
status_label.pack(fill="x", side="bottom")

root.mainloop()