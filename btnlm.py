import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ==========================================
# CÁC HÀM HÌNH HỌC PHỤ TRỢ CHO HỌA ĐỒ VECTOR
# ==========================================

def line_intersection(p1, d1, p2, d2):
    """
    TÌM GIAO ĐIỂM CỦA 2 ĐƯỜNG THẲNG TRÊN MẶT PHẲNG HỌA ĐỒ (QUY TẮC CRAMER)
    - Đường 1: đi qua điểm p1 với vector chỉ phương d1 (r1 = p1 + t * d1)
    - Đường 2: đi qua điểm p2 với vector chỉ phương d2 (r2 = p2 + u * d2)
    """
    # det là định thức chính D của hệ phương trình bậc nhất 2 ẩn (t, u)
    # det = d1x * d2y - d1y * d2x (tương đương tích có hướng 2D giữa vector d1 và d2)
    det = d1[0] * d2[1] - d1[1] * d2[0]
    
    # Nếu det xấp xỉ bằng 0: hai đường thẳng song song hoặc trùng nhau, không có giao điểm duy nhất
    if abs(det) < 1e-9:
        return None
        
    # delta là vector nối gốc p1 sang gốc p2 (delta = p2 - p1)
    delta = p2 - p1
    
    # Tính tham số t bằng quy tắc Cramer: t = D_t / D
    # Đoạn này tìm tỉ lệ bước đi dọc theo vector d1 để chạm tới giao điểm
    t = (delta[0] * d2[1] - delta[1] * d2[0]) / det
    
    # Trả về tọa độ điểm giao nhau: xuất phát từ p1 đi một đoạn t * d1
    return p1 + t * d1


def sim_triangle(p_b, p_c, P_B, P_C, P_target):
    """
    ĐỊNH LÝ ĐỒNG DẠNG TAM GIÁC TRÊN HỌA ĐỒ (Δbce ~ ΔBCE hoặc Δb'c'e' ~ ΔBCE)
    - P_B, P_C, P_target: Tọa độ thực của các khớp trên cơ cấu cơ khí (m)
    - p_b, p_c: Tọa độ đỉnh tương ứng đã tìm được trên họa đồ (m/s hoặc m/s²)
    - Kết quả: Tọa độ đỉnh mục tiêu (m, e hoặc m', e') trên họa đồ
    """
    # 1. Phân tích vector trên cơ cấu thực:
    v_BC = P_C - P_B                  # Vector dọc thanh BC trên cơ cấu
    L_BC_sq = np.dot(v_BC, v_BC)      # Bình phương chiều dài khâu BC: |BC|²
    v_BP = P_target - P_B             # Vector nối từ B tới điểm cần tìm (M hoặc E)
    
    # 2. Tính hai hệ số tỉ lệ đồng dạng:
    # alpha: tỉ lệ hình chiếu dọc theo trục thanh BC (chiếu song song)
    alpha = np.dot(v_BP, v_BC) / L_BC_sq
    # beta: tỉ lệ độ lệch vuông góc khỏi trục thanh BC (chiếu trực giao kèm dấu xét hướng xoay)
    beta = (v_BP[0] * v_BC[1] - v_BP[1] * v_BC[0]) / L_BC_sq
    
    # 3. Dựng điểm tương ứng trên mặt phẳng họa đồ:
    v_bc = p_c - p_b                                  # Vector bc đã có trên họa đồ
    v_bc_perp = np.array([-v_bc[1], v_bc[0]])         # Vector vuông góc với bc (xoay 90 độ ngược CĐH)
    
    # Điểm đích = gốc p_b + (đi dọc theo bc một đoạn alpha) + (rẽ vuông góc một đoạn beta)
    return p_b + alpha * v_bc + beta * v_bc_perp


# ==========================================
# 1. TÍNH VỊ TRÍ HÌNH HỌC TẠI MỘT GÓC BẤT KỲ
# ==========================================

def compute_position_at_phi2(params, current_phi2_deg):
    """
    BÀI TOÁN VỊ TRÍ CƠ CẤU 4 KHÂU ABCD VÀ ĐỈNH TAM GIÁC BCE
    - params: Từ điển chứa kích thước các thanh: L_AB, L_BC, L_CD, L_AD, L_BM, L_ME
    - current_phi2_deg: Giá trị góc quay phi2 của tay quay khâu 2 (tính bằng độ)
    - Trả về: Tọa độ (x, y) của tất cả 6 điểm A, B, C, D, M, E
    """
    # Lấy kích thước các thanh từ từ điển tham số
    L_AB, L_BC = params['L_AB'], params['L_BC']
    L_BM, L_ME = params['L_BM'], params['L_ME']
    L_CD, L_AD = params['L_CD'], params['L_CD'] 
    L_CD, L_AD = params['L_CD'], params['L_AD']

    # Đổi góc phi2 từ độ (deg) sang radian (rad) để đưa vào hàm lượng giác máy tính
    phi2 = np.radians(current_phi2_deg)

    # 1. Tọa độ hai gối đỡ cố định trên giá:
    xA, yA = 0.0, 0.0        # Khớp A chọn làm gốc tọa độ cố định O(0, 0)
    xD, yD = L_AD, 0.0       # Khớp D nằm cố định trên trục hoành Ox cách A đoạn L_AD

    # 2. Tọa độ khớp quay B (chuyển động tròn quanh A với bán kính L_AB):
    xB = L_AB * np.cos(phi2)
    yB = L_AB * np.sin(phi2)

    # 3. Kiểm tra điều kiện khép kín cơ cấu (Bất đẳng thức tam giác ΔBCD):
    # d_BD là khoảng cách thực tế giữa khớp động B và khớp cố định D
    d_BD = np.hypot(xD - xB, yD - yB)
    if d_BD > (L_BC + L_CD) or d_BD < abs(L_BC - L_CD):
        return None  # Không khép kín được tam giác BCD -> Cơ cấu rơi vào điểm chết / kẹt biên

    # 4. Tìm tọa độ khớp C (Giao điểm hai đường tròn: tâm B bán kính L_BC và tâm D bán kính L_CD):
    # a: khoảng cách hình chiếu từ đỉnh B đến chân đường cao hạ từ C xuống cạnh đáy BD
    a = (L_BC**2 - L_CD**2 + d_BD**2) / (2 * d_BD)
    # h: độ dài chiều cao tam giác hạ từ đỉnh C xuống cạnh đáy BD
    h = np.sqrt(max(0.0, L_BC**2 - a**2))

    # Tọa độ chân đường cao (điểm H nằm ngay trên đoạn nối BD):
    x2 = xB + a * (xD - xB) / d_BD
    y2 = yB + a * (yD - yB) / d_BD

    # Tọa độ khớp C: từ chân đường cao bước vuông góc một đoạn đúng bằng h
    xC = x2 - h * (yD - yB) / d_BD
    yC = y2 + h * (xD - xB) / d_BD

    # 5. Xác định tọa độ hai điểm M và E trên khâu 3:
    # theta3: góc nghiêng của thanh truyền BC so với trục nằm ngang Ox
    theta3 = np.arctan2(yC - yB, xC - xB)
    # u_BC: vector đơn vị chỉ phương dọc theo chiều thanh BC (hướng từ B sang C)
    u_BC = np.array([np.cos(theta3), np.sin(theta3)])
    # n_BC: vector đơn vị pháp tuyến vuông góc với thanh BC
    n_BC = np.array([np.sin(theta3), -np.cos(theta3)])

    # Điểm M nằm trên thân thanh BC: từ B đi dọc theo vector u_BC một khoảng L_BM
    xM, yM = np.array([xB, yB]) + L_BM * u_BC
    # Điểm E nhô vuông góc khỏi thanh BC: từ M đi theo vector n_BC một khoảng L_ME
    xE, yE = np.array([xM, yM]) + L_ME * n_BC

    # Gom tọa độ 6 khớp vào từ điển trả về
    return {
        'A': np.array([xA, yA]), 'B': np.array([xB, yB]),
        'C': np.array([xC, yC]), 'D': np.array([xD, yD]),
        'M': np.array([xM, yM]), 'E': np.array([xE, yE])
    }


# ==========================================
# 2.PHƯƠNG PHÁP HỌA ĐỒ VECTOR
# ==========================================

def compute_kinematics_polygon(params):
    """
    XÂY DỰNG HỌA ĐỒ VECTOR VẬN TỐC & GIA TỐC VÀ TÍNH ĐỘ LỚN CỤ THỂ
    - Phương trình vận tốc: v_C = v_B + v_CB
    - Phương trình gia tốc: a_C^n + a_C^t = a_B^n + a_B^t + a_CB^n + a_CB^t
    """
    # Lấy vị trí 6 khớp tại góc phi2 hiện tại
    pts = compute_position_at_phi2(params, params['phi2_deg'])
    if pts is None:
        raise ValueError("Cơ cấu không khép kín được!")

    A, B, C, D = pts['A'], pts['B'], pts['C'], pts['D']
    M, E = pts['M'], pts['E']

    L_AB, L_BC = params['L_AB'], params['L_BC']
    L_CD, L_AD = params['L_CD'], params['L_AD']

    # Quy ước dấu chiều quay: CCW (ngược CĐH) là (+), CW (cùng CĐH) là (-)
    omega2 = -params['omega2'] if params['cw_omega2'] else params['omega2']
    eps2 = -params['eps2'] if params['cw_eps2'] else params['eps2']

    # ---------------------------------------------------------
    # BƯỚC 1: HỌA ĐỒ VẬN TỐC & TÍNH ĐỘ LỚN VẬN TỐC DÀI CỤ THỂ
    # ---------------------------------------------------------
    # Chọn gốc tọa độ làm cực vận tốc pv = (0, 0)
    pv = np.array([0.0, 0.0])

    # 1. Tính vector vận tốc điểm B: v_B vuông góc với thanh dẫn AB (v_B = w2 x r_AB)
    r_AB = B - A
    vB = np.array([-omega2 * r_AB[1], omega2 * r_AB[0]])
    vB_mag = float(np.linalg.norm(vB))  # Độ lớn cụ thể |vB| = w2 * L_AB (m/s)

    # 2. Phương của vector vận tốc v_C: vuông góc với thanh CD (C quay quanh D)
    r_DC = C - D
    d_vC = np.array([-r_DC[1], r_DC[0]])
    d_vC /= np.linalg.norm(d_vC)        # Chuẩn hóa thành vector đơn vị chỉ phương

    # 3. Phương của vận tốc tương đối v_CB: vuông góc với thanh BC (C quay quanh B)
    r_BC = C - B
    d_vCB = np.array([-r_BC[1], r_BC[0]])
    d_vCB /= np.linalg.norm(d_vCB)      # Chuẩn hóa thành vector đơn vị chỉ phương

    # 4. Tìm đỉnh c trên họa đồ vận tốc: giao của đường qua pv (phương d_vC) và qua b (phương d_vCB)
    vC = line_intersection(pv, d_vC, vB, d_vCB)
    vC_mag = float(np.linalg.norm(vC))  # Độ lớn cụ thể của vận tốc điểm C: |vC| (m/s)

    # Vector vận tốc tương đối v_CB = v_C - v_B
    vCB = vC - vB
    vCB_mag = float(np.linalg.norm(vCB)) # Độ lớn cụ thể của vận tốc tương đối |vCB| (m/s)

    # 5. Xác định vận tốc góc của các khâu:
    # omega3 = v_CB / L_BC (xác định chiều quay qua tích có hướng 2D)
    omega3 = (r_BC[0] * vCB[1] - r_BC[1] * vCB[0]) / (L_BC**2)
    # omega4 = v_C / L_CD (xác định chiều quay qua tích có hướng 2D)
    omega4 = (r_DC[0] * vC[1] - r_DC[1] * vC[0]) / (L_CD**2)

    # 6. Dựng vector vận tốc điểm M và E bằng tam giác đồng dạng vận tốc: Δbce ~ ΔBCE
    vM = sim_triangle(vB, vC, B, C, M)
    vM_mag = float(np.linalg.norm(vM))  # Độ lớn cụ thể của vận tốc điểm M: |vM| (m/s)

    vE = sim_triangle(vB, vC, B, C, E)
    vE_mag = float(np.linalg.norm(vE))  # Độ lớn cụ thể của vận tốc điểm E: |vE| (m/s)

    # Gom các giá trị độ lớn vận tốc cụ thể (m/s) vào từ điển
    vel_mag = {
        'vB': vB_mag,
        'vC': vC_mag,
        'vCB': vCB_mag,
        'vM': vM_mag,
        'vE': vE_mag
    }

    # ---------------------------------------------------------
    # BƯỚC 2: HỌA ĐỒ GIA TỐC & TÍNH ĐỘ LỚN GIA TỐC CỤ THỂ
    # ---------------------------------------------------------
    # Chọn gốc tọa độ làm cực gia tốc pi_a = (0, 0)
    pi_a = np.array([0.0, 0.0])

    # 1. Gia tốc của khớp B:
    # a_B^n: gia tốc pháp tuyến hướng từ B về tâm quay A (-w2² * r_AB)
    aB_n = -(omega2**2) * r_AB
    # a_B^t: gia tốc tiếp tuyến vuông góc với thanh AB (eps2 x r_AB)
    aB_t = np.array([-eps2 * r_AB[1], eps2 * r_AB[0]])
    # Gia tốc toàn phần của điểm B: a_B = a_B^n + a_B^t
    aB = aB_n + aB_t
    aB_mag = float(np.linalg.norm(aB))  # Độ lớn cụ thể gia tốc điểm B: |aB| (m/s²)

    # 2. Gia tốc pháp tuyến của các khâu (hướng về tâm quay tương ứng):
    # a_C^n: hướng từ C về tâm gối D với độ lớn w4² * L_CD
    aC_n = -(omega4**2) * r_DC
    # a_CB^n: hướng từ C về khớp B với độ lớn w3² * L_BC
    aCB_n = -(omega3**2) * r_BC

    # 3. Xác định hai điểm mút để kẻ các phương tiếp tuyến vuông góc:
    # Điểm k1 (nhánh C): xuất phát từ cực pi_a cộng thêm thành phần pháp tuyến a_C^n
    k1 = pi_a + aC_n
    d_aC_t = d_vC        # Phương tiếp tuyến a_C^t vuông góc CD (cùng phương với v_C)

    # Điểm k2 (nhánh B): xuất phát từ đỉnh b' (aB) cộng thêm thành phần pháp tuyến tương đối a_CB^n
    k2 = aB + aCB_n
    d_aCB_t = d_vCB      # Phương tiếp tuyến a_CB^t vuông góc BC (cùng phương với v_CB)

    # 4. Tìm đỉnh c' trên họa đồ gia tốc: giao điểm của 2 tia tiếp tuyến qua k1 và k2
    aC = line_intersection(k1, d_aC_t, k2, d_aCB_t)
    aC_mag = float(np.linalg.norm(aC))  # Độ lớn cụ thể của gia tốc điểm C: |aC| (m/s²)

    # Gia tốc tương đối toàn phần của C quanh B: a_CB = a_C - a_B
    aCB = aC - aB
    aCB_mag = float(np.linalg.norm(aCB)) # Độ lớn gia tốc tương đối |aCB| (m/s²)

    # 5. Xác định gia tốc góc các khâu:
    # Tách thành phần tiếp tuyến a_CB^t để tính eps3 = a_CB^t / L_BC
    aCB_t = aCB - aCB_n
    eps3 = (r_BC[0] * aCB_t[1] - r_BC[1] * aCB_t[0]) / (L_BC**2)

    # Tách thành phần tiếp tuyến a_C^t để tính eps4 = a_C^t / L_CD
    aC_t = aC - aC_n
    eps4 = (r_DC[0] * aC_t[1] - r_DC[1] * aC_t[0]) / (L_CD**2)

    # 6. Dựng vector gia tốc điểm M và E bằng tam giác đồng dạng gia tốc: Δb'c'e' ~ ΔBCE
    aM = sim_triangle(aB, aC, B, C, M)
    aM_mag = float(np.linalg.norm(aM))  # Độ lớn cụ thể gia tốc điểm M: |aM| (m/s²)

    aE = sim_triangle(aB, aC, B, C, E)
    aE_mag = float(np.linalg.norm(aE))  # Độ lớn cụ thể gia tốc điểm E: |aE| (m/s²)

    # Gom các giá trị độ lớn gia tốc cụ thể (m/s²) vào từ điển
    acc_mag = {
        'aB': aB_mag,
        'aC': aC_mag,
        'aCB': aCB_mag,
        'aM': aM_mag,
        'aE': aE_mag
    }

    # Trả về kết quả hoàn chỉnh gồm các vector và các con số độ lớn cụ thể
    return {
        'params': params,
        'points': pts,
        'omega': {'omega3': omega3, 'omega4': omega4},
        'eps': {'eps3': eps3, 'eps4': eps4},
        'vel': {'vB': vB, 'vC': vC, 'vM': vM, 'vE': vE},
        'acc': {'aB': aB, 'aC': aC, 'aM': aM, 'aE': aE},
        'vel_mag': vel_mag,  # <-- Độ lớn vận tốc dài các điểm
        'acc_mag': acc_mag   # <-- Độ lớn gia tốc dài các điểm
    }
    # -------------------------------------------------------------------------
    # ĐÓNG GÓI DỮ LIỆU ĐỂ TRUYỀN SANG PHẦN GIAO DIỆN HIỂN THỊ
    # -------------------------------------------------------------------------
    return {
        'params': params,
        'points': pts,
        'omega': {'omega3': omega3, 'omega4': omega4},
        'eps': {'eps3': eps3, 'eps4': eps4},
        'vel': {'vB': vB, 'vC': vC, 'vM': vM, 'vE': vE},
        'acc': {'aB': aB, 'aC': aC, 'aM': aM, 'aE': aE}
    }
# ==========================================
# 3. GIAO DIỆN HMI KÈM ANIMATION
# ==========================================
class DynamicMechanismApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PHẦN MỀM TÍNH TOÁN ĐỘNG HỌC THEO PHƯƠNG PHÁP HỌA ĐỒ VECTOR")
        self.geometry("1300x800")
        self.configure(bg="#F3F4F6")

        self.current_data = None
        self.current_tab = "mech"

        self.is_animating = False
        self.anim_phi2 = 10.0
        self.step_dir = -1.0

        self.create_widgets()
        self.load_preset(1)

    def create_widgets(self):
        left_container = tk.Frame(self, width=420, bg="#FFFFFF", relief="groove", bd=1)
        left_container.pack(side="left", fill="y", padx=10, pady=10)

        canvas_left = tk.Canvas(left_container, bg="#FFFFFF", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas_left.yview)
        self.scroll_frame = tk.Frame(canvas_left, bg="#FFFFFF", padx=10, pady=10)

        self.scroll_frame.bind("<Configure>", lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all")))
        canvas_left.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas_left.configure(yscrollcommand=scrollbar.set)

        canvas_left.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Label(self.scroll_frame, text="THÔNG SỐ CƠ CẤU", font=("Arial", 14, "bold"), 
                 fg="#1E3A8A", bg="#FFFFFF").pack(anchor="w", pady=(0, 10))

        preset_frame = tk.Frame(self.scroll_frame, bg="#FFFFFF")
        preset_frame.pack(fill="x", pady=(0, 10))
        tk.Button(preset_frame, text="Mẫu PA 1", bg="#E0E7FF", fg="#3730A3", font=("Arial", 9, "bold"),
                  command=lambda: self.load_preset(1), relief="flat").pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(preset_frame, text="Mẫu PA 2", bg="#DCFCE7", fg="#166534", font=("Arial", 9, "bold"),
                  command=lambda: self.load_preset(2), relief="flat").pack(side="left", expand=True, fill="x", padx=2)

        input_box = tk.LabelFrame(self.scroll_frame, text="Kích thước các khâu (m)", font=("Arial", 10, "bold"), 
                                  bg="#FFFFFF", padx=10, pady=5)
        input_box.pack(fill="x", pady=5)

        self.entries = {}
        fields = [
            ("L_AB (Khâu 2)", "L_AB"),
            ("L_BC (Khâu 3)", "L_BC"),
            ("L_CD (Khâu 4)", "L_CD"),
            ("L_AD (Giá)", "L_AD"),
            ("L_BM", "L_BM"),
            ("L_ME (⊥ BC)", "L_ME"),
        ]
        for idx, (lbl, key) in enumerate(fields):
            tk.Label(input_box, text=lbl+":", bg="#FFFFFF", anchor="w").grid(row=idx, column=0, sticky="w", pady=2)
            ent = tk.Entry(input_box, width=10, font=("Consolas", 10))
            ent.grid(row=idx, column=1, sticky="e", pady=2, padx=5)
            self.entries[key] = ent

        motion_box = tk.LabelFrame(self.scroll_frame, text="Vị trí & Vận tốc khâu dẫn 2", font=("Arial", 10, "bold"), 
                                   bg="#FFFFFF", padx=10, pady=5)
        motion_box.pack(fill="x", pady=5)

        tk.Label(motion_box, text="Góc φ₂ (độ):", bg="#FFFFFF").grid(row=0, column=0, sticky="w", pady=2)
        self.entries["phi2_deg"] = tk.Entry(motion_box, width=10, font=("Consolas", 10))
        self.entries["phi2_deg"].grid(row=0, column=1, sticky="e", pady=2, padx=5)

        tk.Label(motion_box, text="Vận tốc góc ω₂:", bg="#FFFFFF").grid(row=1, column=0, sticky="w", pady=2)
        self.entries["omega2"] = tk.Entry(motion_box, width=10, font=("Consolas", 10))
        self.entries["omega2"].grid(row=1, column=1, sticky="e", pady=2, padx=5)
        
        self.cw_w2 = tk.BooleanVar(value=True)
        tk.Checkbutton(motion_box, text="Cùng CĐH", variable=self.cw_w2, bg="#FFFFFF").grid(row=1, column=2, sticky="w")

        tk.Label(motion_box, text="Gia tốc góc ε₂:", bg="#FFFFFF").grid(row=2, column=0, sticky="w", pady=2)
        self.entries["eps2"] = tk.Entry(motion_box, width=10, font=("Consolas", 10))
        self.entries["eps2"].grid(row=2, column=1, sticky="e", pady=2, padx=5)

        self.cw_eps2 = tk.BooleanVar(value=True)
        tk.Checkbutton(motion_box, text="Cùng CĐH", variable=self.cw_eps2, bg="#FFFFFF").grid(row=2, column=2, sticky="w")

        ctrl_frame = tk.Frame(self.scroll_frame, bg="#FFFFFF")
        ctrl_frame.pack(fill="x", pady=10)

        tk.Button(ctrl_frame, text="⚡ CẬP NHẬT TĨNH", font=("Arial", 10, "bold"),
                  bg="#2563EB", fg="white", height=2, relief="flat", cursor="hand2",
                  command=self.on_calculate).pack(fill="x", pady=2)

        self.btn_animate = tk.Button(ctrl_frame, text="▶ CHẠY MÔ PHỎNG CHUYỂN ĐỘNG", font=("Arial", 11, "bold"),
                                     bg="#8B5CF6", fg="white", activebackground="#7C3AED", height=2, relief="flat",
                                     cursor="hand2", command=self.toggle_animation)
        self.btn_animate.pack(fill="x", pady=4)

        btn_grid = tk.Frame(self.scroll_frame, bg="#FFFFFF")
        btn_grid.pack(fill="x", pady=5)
        
        tk.Button(btn_grid, text="1. Lược đồ Cơ cấu", bg="#3B82F6", fg="white", relief="flat",
                  command=lambda: self.switch_tab("mech")).pack(fill="x", pady=2)
        tk.Button(btn_grid, text="2. Họa đồ Vận tốc", bg="#10B981", fg="white", relief="flat",
                  command=lambda: self.switch_tab("vel")).pack(fill="x", pady=2)
        tk.Button(btn_grid, text="3. Họa đồ Gia tốc", bg="#F59E0B", fg="white", relief="flat",
                  command=lambda: self.switch_tab("acc")).pack(fill="x", pady=2)

        tk.Label(self.scroll_frame, text="KẾT QUẢ TÍNH TOÁN (HỌA ĐỒ):", font=("Arial", 10, "bold"), 
                 fg="#374151", bg="#FFFFFF").pack(anchor="w", pady=(10, 2))
        self.txt_res = tk.Text(self.scroll_frame, font=("Consolas", 9), height=14, width=38,
                               bg="#F9FAFB", fg="#111827", relief="solid", bd=1)
        self.txt_res.pack(fill="x", pady=2)

        right_panel = tk.Frame(self, bg="#F3F4F6")
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(8.5, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_preset(self, pa):
        self.stop_animation()
        for ent in self.entries.values():
            ent.delete(0, tk.END)

        if pa == 1:
            vals = {
                "L_AB": "5.0", "L_BC": "4.5", "L_BM": "2.0", "L_ME": "1.0", "L_CD": "6.0", "L_AD": "3.0",
                "phi2_deg": "10.0", "omega2": "5.0", "eps2": "10.0"
            }
            self.cw_w2.set(True)
            self.cw_eps2.set(True)
        else:
            vals = {
                "L_AB": "5.0", "L_BC": "4.0", "L_BM": "2.5", "L_ME": "1.5", "L_CD": "6.0", "L_AD": "3.0",
                "phi2_deg": "20.0", "omega2": "5.0", "eps2": "20.0"
            }
            self.cw_w2.set(True)
            self.cw_eps2.set(False)

        for k, v in vals.items():
            self.entries[k].insert(0, v)
        
        self.on_calculate()

    def get_input_parameters(self):
        try:
            return {
                'L_AB': float(self.entries['L_AB'].get()),
                'L_BC': float(self.entries['L_BC'].get()),
                'L_BM': float(self.entries['L_BM'].get()),
                'L_ME': float(self.entries['L_ME'].get()),
                'L_CD': float(self.entries['L_CD'].get()),
                'L_AD': float(self.entries['L_AD'].get()),
                'phi2_deg': float(self.entries['phi2_deg'].get()),
                'omega2': float(self.entries['omega2'].get()),
                'eps2': float(self.entries['eps2'].get()),
                'cw_omega2': self.cw_w2.get(),
                'cw_eps2': self.cw_eps2.get()
            }
        except ValueError:
            messagebox.showerror("Lỗi dữ liệu", "Vui lòng nhập đúng định dạng số cho tất cả các ô!")
            return None

    def on_calculate(self):
        params = self.get_input_parameters()
        if params is None:
            return
        try:
            self.current_data = compute_kinematics_polygon(params)
            self.anim_phi2 = params['phi2_deg']
            self.render_view()
        except Exception as e:
            messagebox.showerror("Lỗi hình học cơ cấu", str(e))

    def toggle_animation(self):
        if self.is_animating:
            self.stop_animation()
        else:
            self.start_animation()

    def start_animation(self):
        params = self.get_input_parameters()
        if params is None:
            return
        self.is_animating = True
        self.current_tab = "mech"
        self.btn_animate.config(text="⏸ TẠM DỪNG MÔ PHỎNG", bg="#EF4444")
        
        self.step_dir = -1.0 if params['cw_omega2'] else 1.0
        self.anim_phi2 = params['phi2_deg']
        self.animate_loop()

    def stop_animation(self):
        self.is_animating = False
        self.btn_animate.config(text="▶ CHẠY MÔ PHỎNG CHUYỂN ĐỘNG", bg="#8B5CF6")

    def animate_loop(self):
        if not self.is_animating:
            return

        params = self.get_input_parameters()
        if params is None:
            self.stop_animation()
            return

        speed_deg = max(0.5, params['omega2'] * 0.4)
        next_phi2 = (self.anim_phi2 + self.step_dir * speed_deg) % 360.0

        pts = compute_position_at_phi2(params, next_phi2)
        if pts is None:
            self.step_dir = -self.step_dir
            next_phi2 = (self.anim_phi2 + self.step_dir * speed_deg) % 360.0
            pts = compute_position_at_phi2(params, next_phi2)

        if pts is not None:
            self.anim_phi2 = next_phi2
            self.entries["phi2_deg"].delete(0, tk.END)
            self.entries["phi2_deg"].insert(0, f"{self.anim_phi2:.1f}")
            self.draw_mechanism_pts(pts, self.anim_phi2, params)

        self.after(25, self.animate_loop)

    def switch_tab(self, tab_name):
        if self.is_animating and tab_name != "mech":
            self.stop_animation()
        self.current_tab = tab_name
        self.render_view()

    def render_view(self):
        if not self.current_data:
            return
        
        d = self.current_data
        cw_ccw = lambda val: "Cùng CĐH (CW)" if val < 0 else "Ngược CĐH (CCW)"

        if self.current_tab == "mech":
            self.draw_mechanism_pts(d['points'], d['params']['phi2_deg'], d['params'])
            self.txt_res.delete("1.0", tk.END)
            
            # 1. Phân tích cấu trúc cơ cấu & Xếp loại
            self.txt_res.insert(tk.END, "=== 1. PHÂN TÍCH CẤU TRÚC CƠ CẤU ===\n")
            self.txt_res.insert(tk.END, "• Số khâu động (n): 3 (Khâu 2, 3, 4)\n")
            self.txt_res.insert(tk.END, "• Khâu dẫn: Khâu 2 (thanh AB)\n")
            self.txt_res.insert(tk.END, "• Khâu bị dẫn: Khâu 3 (BCE) & Khâu 4 (CD)\n")
            self.txt_res.insert(tk.END, "• Số khớp thấp (p5): 4 (Khớp A, B, C, D)\n")
            self.txt_res.insert(tk.END, "• Số khớp cao (p4): 0\n\n")
            
            self.txt_res.insert(tk.END, "• Bậc tự do :\n")
            self.txt_res.insert(tk.END, "  W = 3*n - 2*p5 - p4\n")
            self.txt_res.insert(tk.END, "  W = 3*3 - 2*4 - 0 = 1\n")
            self.txt_res.insert(tk.END, "  => Cơ cấu có 1 BẬC TỰ DO.\n\n")
            
            self.txt_res.insert(tk.END, "• Tách nhóm tĩnh định (Assur):\n")
            self.txt_res.insert(tk.END, "  - Cơ cấu cơ sở: Khâu 2 nối giá 1 (Loại I)\n")
            self.txt_res.insert(tk.END, "  - Nhóm Assur: 2 khâu (3, 4) và 3 khớp (B, C, D)\n")
            self.txt_res.insert(tk.END, "    -> Nhóm Assur LOẠI II.\n\n")
            
            self.txt_res.insert(tk.END, "• XẾP LOẠI: Cơ cấu thuộc LOẠI II.\n")
            self.txt_res.insert(tk.END, "====================================\n\n")
            
            # 2. Hiển thị tọa độ các khớp
            self.txt_res.insert(tk.END, "=== TỌA ĐỘ CÁC KHỚP ===\n")
            for k, v in d['points'].items():
                self.txt_res.insert(tk.END, f"Điểm {k:2s}: ({v[0]:6.3f}, {v[1]:6.3f}) m\n")

        elif self.current_tab == "vel":
            self.draw_velocity(d)
            w = d['omega']
            v = d['vel']
            self.txt_res.delete("1.0", tk.END)
            self.txt_res.insert(tk.END, "=== KẾT QUẢ VẬN TỐC (HỌA ĐỒ) ===\n")
            self.txt_res.insert(tk.END, f"ω₃: {abs(w['omega3']):.4f} rad/s\n    -> {cw_ccw(w['omega3'])}\n")
            self.txt_res.insert(tk.END, f"ω₄: {abs(w['omega4']):.4f} rad/s\n    -> {cw_ccw(w['omega4'])}\n\n")
            self.txt_res.insert(tk.END, "Độ lớn vận tốc dài:\n")
            for k in ['vB', 'vC', 'vM', 'vE']:
                self.txt_res.insert(tk.END, f"|{k}| = {np.linalg.norm(v[k]):.4f} m/s\n")

        elif self.current_tab == "acc":
            self.draw_acceleration(d)
            eps = d['eps']
            a = d['acc']
            self.txt_res.delete("1.0", tk.END)
            self.txt_res.insert(tk.END, "=== KẾT QUẢ GIA TỐC (HỌA ĐỒ) ===\n")
            self.txt_res.insert(tk.END, f"ε₃: {abs(eps['eps3']):.4f} rad/s²\n    -> {cw_ccw(eps['eps3'])}\n")
            self.txt_res.insert(tk.END, f"ε₄: {abs(eps['eps4']):.4f} rad/s²\n    -> {cw_ccw(eps['eps4'])}\n\n")
            self.txt_res.insert(tk.END, "Độ lớn gia tốc dài:\n")
            for k in ['aB', 'aC', 'aM', 'aE']:
                self.txt_res.insert(tk.END, f"|{k}| = {np.linalg.norm(a[k]):.4f} m/s²\n")

    def draw_mechanism_pts(self, pts, phi_deg, params):
        self.ax.clear()
        xA, yA = pts['A']
        xB, yB = pts['B']
        xC, yC = pts['C']
        xD, yD = pts['D']
        xM, yM = pts['M']
        xE, yE = pts['E']

        self.ax.plot([xA, xD], [yA, yD], 'k--', lw=2, label='Giá AD')
        self.ax.plot([xA, xB], [yA, yB], 'b-o', lw=3, label='Khâu 2 (AB)')
        self.ax.plot([xB, xC], [yB, yC], 'g-o', lw=3, label='Khâu 3 (BC)')
        self.ax.plot([xD, xC], [yD, yC], 'm-o', lw=3, label='Khâu 4 (CD)')
        self.ax.plot([xM, xE], [yM, yE], 'r-o', lw=2.5, label='Thanh ME ⊥ BC')

        self.ax.plot(xA, yA, 'ks', markersize=8)
        self.ax.plot(xD, yD, 'ks', markersize=8)

        for name, p in pts.items():
            self.ax.annotate(f' {name}', (p[0], p[1]), textcoords="offset points", 
                             xytext=(8, 6), fontsize=12, fontweight='bold')

        self.ax.set_xlim(-params['L_AB'] - 1.5, params['L_AD'] + params['L_CD'] + 1.5)
        self.ax.set_ylim(-params['L_AB'] - 2.0, params['L_AB'] + params['L_BC'] + 1.5)

        direction_lbl = "Cùng CĐH" if params['cw_omega2'] else "Ngược CĐH"
        self.ax.set_title(f"Lược đồ cơ cấu - φ₂ = {phi_deg:.1f}° | Chiều quay: {direction_lbl}", 
                          fontsize=13, fontweight='bold', pad=12)
        self.ax.set_xlabel("X (m)", fontsize=11)
        self.ax.set_ylabel("Y (m)", fontsize=11)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_aspect('equal', 'box')
        
        self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

    def draw_velocity(self, d):
        self.ax.clear()
        v = d['vel']
        
        self.ax.plot(0, 0, 'ko', markersize=7)
        self.ax.annotate(" pv", (0, 0), fontsize=12, fontweight='bold')

        vel_points = {'b': v['vB'], 'c': v['vC'], 'm': v['vM'], 'e': v['vE']}
        colors = {'b': 'blue', 'c': 'magenta', 'm': 'green', 'e': 'red'}

        for name, vec in vel_points.items():
            self.ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
                           color=colors[name], width=0.007, label=f"v_{name.upper()}")
            self.ax.annotate(f" {name}", (vec[0], vec[1]), textcoords="offset points", 
                             xytext=(8, 6), fontsize=12, fontweight='bold', color=colors[name])

        self.ax.plot([v['vB'][0], v['vC'][0]], [v['vB'][1], v['vC'][1]], 'k:', lw=1.2)
        self.ax.plot([v['vC'][0], v['vE'][0]], [v['vC'][1], v['vE'][1]], 'k:', lw=1.2)
        self.ax.plot([v['vB'][0], v['vE'][0]], [v['vB'][1], v['vE'][1]], 'k:', lw=1.2)

        all_vx = [vec[0] for vec in vel_points.values()] + [0.0]
        all_vy = [vec[1] for vec in vel_points.values()] + [0.0]
        span_y = max(all_vy) - min(all_vy)
        pad_x = span_y * 0.25
        self.ax.set_xlim(min(all_vx) - pad_x, max(all_vx) + pad_x)
        self.ax.set_ylim(min(all_vy) - 5, max(all_vy) + 5)

        self.ax.set_title("Họa đồ Vận tốc (Cực pv tại gốc)", fontsize=13, fontweight='bold', pad=12)
        self.ax.set_xlabel("Vx (m/s)", fontsize=11)
        self.ax.set_ylabel("Vy (m/s)", fontsize=11)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_aspect('auto')

        self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

    def draw_acceleration(self, d):
        self.ax.clear()
        a = d['acc']

        self.ax.plot(0, 0, 'ko', markersize=7)
        self.ax.annotate(" π_a", (0, 0), fontsize=12, fontweight='bold')

        acc_points = {"b'": a['aB'], "c'": a['aC'], "m'": a['aM'], "e'": a['aE']}
        colors = {"b'": 'blue', "c'": 'magenta', "m'": 'green', "e'": 'red'}

        for name, vec in acc_points.items():
            self.ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
                           color=colors[name], width=0.007, label=f"a_{name[0].upper()}")
            self.ax.annotate(f" {name}", (vec[0], vec[1]), textcoords="offset points", 
                             xytext=(8, 6), fontsize=12, fontweight='bold', color=colors[name])

        self.ax.plot([a['aB'][0], a['aC'][0]], [a['aB'][1], a['aC'][1]], 'k:', lw=1.2)
        self.ax.plot([a['aC'][0], a['aE'][0]], [a['aC'][1], a['aE'][1]], 'k:', lw=1.2)
        self.ax.plot([a['aB'][0], a['aE'][0]], [a['aB'][1], a['aE'][1]], 'k:', lw=1.2)

        all_ax = [vec[0] for vec in acc_points.values()] + [0.0]
        all_ay = [vec[1] for vec in acc_points.values()] + [0.0]
        pad_x = (max(all_ax) - min(all_ax)) * 0.15
        pad_y = (max(all_ay) - min(all_ay)) * 0.15

        self.ax.set_xlim(min(all_ax) - pad_x, max(all_ax) + pad_x)
        self.ax.set_ylim(min(all_ay) - pad_y, max(all_ay) + pad_y)

        self.ax.set_title("Họa đồ Gia tốc (Cực π_a tại gốc)", fontsize=13, fontweight='bold', pad=12)
        self.ax.set_xlabel("Ax (m/s²)", fontsize=11)
        self.ax.set_ylabel("Ay (m/s²)", fontsize=11)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_aspect('auto')

        self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = DynamicMechanismApp()
    app.mainloop()