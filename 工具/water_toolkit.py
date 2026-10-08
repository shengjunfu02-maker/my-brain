# -*- coding: utf-8 -*-
"""
水质检测照片处理工具包 water_toolkit.py
用法（在本资料包内，路径相对于 随身资料/ 根目录）:
    exec(open("工具/water_toolkit.py", encoding="utf-8").read())
    r = read_photo("照片存档/水样照片_20261007_001.jpg", {"pH": (x1,y1,x2,y2), "氨氮": (...), "亚盐": (...)})
流程: 旋转摆正 → 白平衡校正 → 色调族采样 → 锚点换算 → ⚠️规则
依赖: PIL(Pillow) / numpy / matplotlib
"""
from PIL import Image
import re
import numpy as np
import matplotlib.colors as mcolors

# ---------------- 1. 图片旋转 ----------------
def rotate_tubes_down(fname, out=None, direction="cw"):
    """手持横拍图转正：默认顺时针90°使管口朝下；已是竖图则原样返回。
    返回 PIL.Image（竖图）。out 不为 None 时保存。"""
    img = Image.open(fname).convert("RGB")
    w, h = img.size
    if w > h:  # 横图 -> 顺时针90°(PIL ROTATE_270)
        img = img.transpose(Image.ROTATE_270)
    if out:
        img.save(out, quality=95)
    return img

# ---------------- 2. 白平衡校正 ----------------
def wb_correct(img):
    """用画面内白色像素（水印白字/白卡纸/白桶）做白平衡校正。
    返回 (校正后数组, 是否成功)。"""
    hsv = mcolors.rgb_to_hsv(img/255.)
    white = (hsv[:,:,1] < 0.15) & (hsv[:,:,2] > 0.85)
    if white.sum() < 500:
        return img, False
    mean_rgb = img[white].mean(axis=0)
    gray = mean_rgb.mean()
    return np.clip(img * (gray/mean_rgb), 0, 255), True

# ---------------- 3. 液体采样 ----------------
HF = {"pH": [(0.16,0.46)], "氨氮": [(0.03,0.20)], "亚盐": [(0.78,0.99),(0.00,0.05)]}
ANCHORS_HI_NH3 = [(0.11, 0.60, "2"), (0.099, 0.65, "3"), (0.07, 0.75, "5"),
                  (0.045, 0.80, "8"), (0.02, 0.70, "15")]
def _base_param(p):
    return re.sub(r"[（(].*?[)）]", "", p).strip()

def sample_liquid(src, box, param, s_thr=0.20):
    """WB校正后按色调族过滤，取液体中位色。
    src: 路径或图像数组; box: (x1,y1,x2,y2); param: pH/氨氮/亚盐
    返回 dict: rgb/hsv/n/wb；近无色返回 hsv=None"""
    if isinstance(src, str):
        img = np.array(Image.open(src).convert("RGB")).astype(float)
    else:
        img = src.astype(float)
    img, ok = wb_correct(img)
    x1,y1,x2,y2 = box
    crop = img[y1:y2, x1:x2].reshape(-1,3)
    hsv = mcolors.rgb_to_hsv(crop/255.)
    m = (hsv[:,1] > s_thr) & (hsv[:,2] > 0.30)
    hue_ok = np.zeros(len(hsv), bool)
    for lo, hi in HF[param]:
        hue_ok |= (hsv[:,0] >= lo) & (hsv[:,0] <= hi)
    mask = m & hue_ok
    if mask.sum() < 60:
        mask = (hsv[:,1] > 0.10) & hue_ok
    if mask.sum() < 30:
        return {"rgb": None, "hsv": None, "n": int(mask.sum()), "wb": ok}
    sel = crop[mask]
    q = (sel // 12).astype(int)          # RGB量化分箱(每通道~21档)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    dom = sel[(q == keys[counts.argmax()]).all(axis=1)]
    med = np.median(dom, axis=0)          # 取面积最大档内中位色作代表色
    return {"rgb": med, "hsv": mcolors.rgb_to_hsv(med/255.), "n": int(mask.sum()), "wb": ok}

# ---------------- 4. 锚点换算表（侧视、WB校正后） ----------------
# 格式: (hue, sat, 读数)
ANCHORS = {
    "pH":   [(0.23, 0.60, "7.0"), (0.33, 0.55, "7.2"), (0.42, 0.60, "7.4")],
    "氨氮": [(0.175, 0.40, "0.20"), (0.141, 0.65, "0.40"), (0.121, 0.65, "0.50"),
             (0.100, 0.70, "0.60"), (0.090, 0.75, "0.90")],
    "亚盐": [(0.90, 0.35, "0.10"), (0.90, 0.45, "0.15"), (0.897, 0.48, "0.18"),
             (0.90, 0.62, "0.20"), (0.915, 0.83, "0.25"), (0.944, 0.97, "0.28")],
}
NO_COLOR = {"pH": None, "氨氮": "0.20", "亚盐": "0.05"}   # 近无色按低档

def hsv_to_value(param, hsv):
    """锚点最近邻换算。返回 (读数字符串, 备注)。"""
    if hsv is None:
        v0 = NO_COLOR[_base_param(param)]
        return ("5" if ("高量程" in param and v0) else v0), "近无色"
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    hi = "高量程" in param
    anchors = ANCHORS_HI_NH3 if hi else ANCHORS[_base_param(param)]
    note = ""
    if param == "亚盐" and 0.80 <= h <= 0.88:
        note = "偏紫(背景干扰),按明度定档"   # 红管遇蓝背景拍偏紫
    best, bd = None, 1e9
    for ah, a_s, val in anchors:
        dh = min(abs(h-ah), 1-abs(h-ah))
        w_s = 0.0 if hi else 0.5           # 高量程只看色相；全局色相为主
        d = (3*dh)**2 + (w_s*(s-a_s))**2
        if d < bd:
            bd, best = d, val
    # 亚盐超量程外推: 按0.01-0.3档位的显色规律延伸(色相漂向纯红为主, 明度为辅, 饱和封顶),
    # 现场标定: 爆表≈0.6, 最深纯红≈1.0
    if _base_param(param) == "亚盐" and best == "0.28" and (s > 0.985 or h > 0.965 or h < 0.015):
        h_eff = h if h >= 0.5 else h + 1.0            # 色相绕回处理
        hue_x = min(1.0, max(0.0, (h_eff - 0.944) / 0.056))
        dark_x = min(1.0, max(0.0, (0.98 - v) / 0.13))  # 越暗越深(明度反向)
        sat_x = min(1.0, max(0.0, (s - 0.97) / 0.03))
        x = 0.40 * hue_x + 0.35 * dark_x + 0.10 * sat_x
        val = min(1.0, 0.30 + 0.70 * x)
        return f"{val:.1f}", (note + ";" if note else "") + f"超量程估值({val:.1f}),鱼易浮头"
    return best, note

# ---------------- 5. ⚠️ 规则 ----------------
def need_warn(param, value_str):
    """亚盐>=0.3(或接近,如0.28)；氨氮>=5。"""
    try:
        v = float(value_str)
    except (TypeError, ValueError):
        return False
    p = _base_param(param)
    if _base_param(param) == "亚盐" and v >= 0.28: return True
    if p == "氨氮" and v >= 5:    return True
    return False

# ---------------- 6. 一键读图 ----------------
def read_photo(fname, tubes, view="side"):
    """tubes: {param: box}；view: side=手持侧视 / top=俯拍(色偏深)
    返回 {param: {"value","warn","hsv","n","note"}}"""
    out = {}
    for param, box in tubes.items():
        r = sample_liquid(fname, box, _base_param(param))
        val, note = hsv_to_value(param, r["hsv"])
        if view == "top" and param == "pH" and val:
            note = (note + ";" if note else "") + "俯拍偏深约0.1-0.2"
        out[param] = {"value": val, "warn": need_warn(param, val),
                      "hsv": None if r["hsv"] is None else tuple(np.round(r["hsv"],3)),
                      "n": r["n"], "note": note}
    return out
