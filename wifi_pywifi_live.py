mport subprocess, time, re, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from collections import defaultdict

# ================== Sabitler ==================
SCAN_INTERVAL = 2.0      # her 2 sn'de bir tarama yap
PLOT_INTERVAL = 10.0     # grafiği sadece 10 sn'de bir yenile
DEFAULT_FILTER = "all"   # başlangıç band filtresi: "all" | "2.4" | "5"

# Kanal listeleri
CHS_24 = np.arange(1, 14)  # 1..13
CHS_5  = np.array([36,40,44,48,52,56,60,64,100,104,108,112,116,120,124,128,132,136,140,149,153,157,161,165])

# ================== Yardımcılar ==================
def percent_to_dbm(p):
    # Windows'ta netsh signal % → yaklaşık dBm
    if p is None:
        return None
    return (p / 2.0) - 100.0  # 100%≈-50dBm, 0%≈-100dBm

def band_of_channel(ch):
    if ch is None:
        return None
    return "2.4 GHz" if 1 <= ch <= 14 else "5 GHz"

def stars_from_dbm(dbm):
    if dbm is None: return "☆☆☆☆☆"
    if dbm >= -50: return "★★★★★"
    if dbm >= -60: return "★★★★☆"
    if dbm >= -70: return "★★★☆☆"
    if dbm >= -80: return "★★☆☆☆"
    if dbm >= -90: return "★☆☆☆☆"
    return "☆☆☆☆☆"

def rating_from_score(score):
    # score küçükse iyi → çok yıldız
    if score <= 0.05: return "★★★★★"
    if score <= 0.10: return "★★★★☆"
    if score <= 0.20: return "★★★☆☆"
    if score <= 0.35: return "★★☆☆☆"
    if score <= 0.50: return "★☆☆☆☆"
    return "☆☆☆☆☆"

def color_for_ssid(ssid):
    palette = cm.get_cmap("tab20").colors
    return palette[abs(hash(ssid)) % len(palette)]

def gaussian_curve(x, mu, sigma, strength_dbm):
    amp = max(0.0, 100 + strength_dbm)  # -50dBm -> güçlü tepe
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) * amp

# ================== netsh okumalar ==================
def scan_networks_raw():
    """
    netsh wlan show networks mode=bssid çıktısını parse eder.
    Döner:
    {
       "SSID_ADI":[
           {"ssid":..., "dbm":..., "channel":..., "band":"2.4 GHz"/"5 GHz"},
           ...
       ],
       ...
    }
    """
    try:
        out = subprocess.check_output(
            ["netsh","wlan","show","networks","mode=bssid"],
            text=True, encoding="utf-8", errors="ignore"
        )
    except Exception:
        return {}

    result = {}
    blocks = re.split(r"\nSSID\s+\d+\s*:\s*", out)[1:]
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        ssid = lines[0].strip() or "Bilinmeyen"
        result.setdefault(ssid, [])
        bssids = re.split(r"BSSID\s+\d+\s*:\s*", block)[1:]
        for seg in bssids:
            m_sig = re.search(r"Signal\s*:\s*(\d+)%", seg)
            m_ch  = re.search(r"Channel\s*:\s*(\d+)", seg)
            if not m_sig or not m_ch:
                continue
            pct  = int(m_sig.group(1))
            dbm  = percent_to_dbm(pct)
            ch   = int(m_ch.group(1))
            band = band_of_channel(ch)
            if dbm is None or band is None:
                continue
            result[ssid].append({
                "ssid": ssid,
                "dbm": dbm,
                "channel": ch,
                "band": band
            })
    return result

def scan_interface_speed_raw():
    """
    netsh wlan show interfaces -> mevcut bağlı SSID + hız
    """
    try:
        out = subprocess.check_output(
            ["netsh","wlan","show","interfaces"],
            text=True, encoding="utf-8", errors="ignore"
        )
    except Exception:
        return {}
    info = {}
    m_ssid = re.search(r"SSID\s*:\s*(.+)", out)
    m_rx   = re.search(r"Receive rate \(Mbps\)\s*:\s*([0-9\.]+)", out)
    m_tx   = re.search(r"Transmit rate \(Mbps\)\s*:\s*([0-9\.]+)", out)
    if m_ssid: info["ssid"] = m_ssid.group(1).strip()
    if m_rx:   info["rx"]   = m_rx.group(1)
    if m_tx:   info["tx"]   = m_tx.group(1)
    return info

# ================== Veri hazırlama ==================
def flatten_aps(nets_by_ssid, band_filter):
    """
    band_filter: "all", "2.4", "5"
    Tek tek AP listesi döner.
    """
    aps = []
    for ssid, lst in nets_by_ssid.items():
        for ap in lst:
            band = ap.get("band")
            if band is None: continue
            if band_filter == "all" \
               or (band_filter == "2.4" and band == "2.4 GHz") \
               or (band_filter == "5"   and band == "5 GHz"):
                aps.append(ap)
    return aps

def best_per_ssid(nets_by_ssid, band_filter):
    """
    Sağ panelde göstermek için her SSID'nin en güçlü AP kaydı.
    """
    rows = []
    for ssid, lst in nets_by_ssid.items():
        valids = []
        for ap in lst:
            band = ap.get("band")
            if ap.get("dbm") is None or ap.get("channel") is None or band is None:
                continue
            if band_filter == "all" \
               or (band_filter == "2.4" and band == "2.4 GHz") \
               or (band_filter == "5"   and band == "5 GHz"):
                valids.append(ap)
        if not valids:
            continue
        best = max(valids, key=lambda a: a["dbm"])
        rows.append(best)
    rows.sort(key=lambda r: r["dbm"], reverse=True)
    return rows

# ================== Kanal doluluk skoru (Best Channel) ==================
def compute_channel_scores(nets_by_ssid):
    """
    Basit bir giriş seviyesi kanal doluluk metriği:
    - Her AP, bulunduğu kanaldaki gürültüye katkıda bulunuyor.
    - Daha güçlü (daha yüksek dBm) ise daha büyük katkı.
    - Komşu kanallara da biraz sızma payı (2.4 GHz'te özellikle).
    Çıktı:
    {
      "2.4": { ch: score, ... },
      "5":   { ch: score, ... }
    }
    Skor küçük -> daha boş -> daha iyi kanal.
    """
    scores_24 = {ch:0.0 for ch in CHS_24}
    scores_5  = {ch:0.0 for ch in CHS_5}

    for ssid, lst in nets_by_ssid.items():
        for ap in lst:
            ch = ap.get("channel")
            dbm = ap.get("dbm")
            band = ap.get("band")
            if ch is None or dbm is None: continue
            # normalize "interference weight": güçlü sinyal daha büyük gürültü
            # ör: -50 dBm -> ağırlık 1.0, -80 dBm -> 0.2
            weight = max(0.05, (100 + dbm)/50.0)  # kaba ölçek

            if band == "2.4 GHz" and ch in scores_24:
                # komşu kanallara da biraz ekle
                for delta, leak in [(-2,0.4),(-1,0.7),(0,1.0),(1,0.7),(2,0.4)]:
                    cc = ch + delta
                    if cc in scores_24:
                        scores_24[cc] += weight * leak

            elif band == "5 GHz":
                # 5 GHz'te kanallar daha ayrık ⇒ sadece kendine bas
                if ch in scores_5:
                    scores_5[ch] += weight * 1.0

    return {"2.4":scores_24, "5":scores_5}

def best_channels_text(scores_dict, top_n=3):
    """
    Skor tablosundan en iyi birkaç kanalı sırala ve ↘ yazı formatına çevir.
    Ayrıca yıldız ver (rating_from_score).
    """
    lines = []
    # 2.4
    s24 = scores_dict["2.4"]
    sorted24 = sorted(s24.items(), key=lambda kv: kv[1])
    take24 = sorted24[:top_n]
    if take24:
        pretty_24 = ", ".join(
            [f"{ch} (skor {val:.2f}) {rating_from_score(val)}"
             for ch,val in take24]
        )
        lines.append(f"2.4 GHz → {pretty_24}")
    # 5
    s5 = scores_dict["5"]
    sorted5 = sorted(s5.items(), key=lambda kv: kv[1])
    take5 = sorted5[:top_n]
    if take5:
        pretty_5 = ", ".join(
            [f"{ch} (skor {val:.2f}) {rating_from_score(val)}"
             for ch,val in take5]
        )
        lines.append(f"5 GHz   → {pretty_5}")
    return lines

# ================== Görsel eksen haritalama ==================
#
# Amaç:
# - 2.4 GHz kanallarını solda genişçe göster
# - 5 GHz kanallarını sağda genişçe göster
# - all modunda iki blok arasında boşluk olsun
#
def get_mapping(band_filter):
    if band_filter == "2.4":
        vis_min, vis_max = 0.0, 100.0
        real_min, real_max = 1.0, 13.0
        scale24 = (vis_max - vis_min)/(real_max - real_min)
        def map_ch(ch, band):
            if band != "2.4 GHz": return None
            return vis_min + (ch-real_min)*scale24
        ticks_x = []; ticks_label=[]
        for ch in CHS_24:
            vx = map_ch(ch,"2.4 GHz")
            ticks_x.append(vx); ticks_label.append(str(ch))
        return {
            "x_min": vis_min, "x_max": vis_max,
            "map_ch": map_ch,
            "ticks_x": ticks_x, "ticks_label": ticks_label,
            "sigma_scale_24": scale24,
            "sigma_scale_5": None,
        }

    if band_filter == "5":
        vis_min, vis_max = 0.0, 100.0
        real_min, real_max = CHS_5.min(), CHS_5.max()
        scale5 = (vis_max - vis_min)/(real_max - real_min)
        def map_ch(ch, band):
            if band != "5 GHz": return None
            return vis_min + (ch-real_min)*scale5
        ticks_x=[]; ticks_label=[]
        for ch in CHS_5:
            vx = map_ch(ch,"5 GHz")
            ticks_x.append(vx); ticks_label.append(str(ch))
        return {
            "x_min": vis_min, "x_max": vis_max,
            "map_ch": map_ch,
            "ticks_x": ticks_x, "ticks_label": ticks_label,
            "sigma_scale_24": None,
            "sigma_scale_5": scale5,
        }

    # all: iki blok
    seg24_min, seg24_max = 0.0, 60.0
    seg5_min,  seg5_max  = 80.0, 180.0
    real24_min, real24_max = 1.0, 13.0
    real5_min,  real5_max  = CHS_5.min(), CHS_5.max()
    scale24 = (seg24_max-seg24_min)/(real24_max-real24_min)
    scale5  = (seg5_max -seg5_min )/(real5_max -real5_min )
    def map_ch(ch, band):
        if band == "2.4 GHz":
            return seg24_min + (ch-real24_min)*scale24
        if band == "5 GHz":
            return seg5_min  + (ch-real5_min)*scale5
        return None
    ticks_x=[]; ticks_label=[]
    for ch in CHS_24:
        vx = map_ch(ch,"2.4 GHz")
        ticks_x.append(vx); ticks_label.append(str(ch))
    for ch in CHS_5:
        vx = map_ch(ch,"5 GHz")
        ticks_x.append(vx); ticks_label.append(str(ch))
    return {
        "x_min": 0.0, "x_max": 180.0,
        "map_ch": map_ch,
        "ticks_x": ticks_x, "ticks_label": ticks_label,
        "sigma_scale_24": scale24,
        "sigma_scale_5": scale5,
    }

# ================== Çizim ==================
def draw_plot(ui):
    """
    Son taranmış veriyi (ui.latest_nets / ui.latest_speed / ui.latest_scores)
    kullanarak grafiği ÇİZER.
    Bu fonksiyon 10 saniyede bir çağrılacak.
    """
    nets_by_ssid = ui.latest_nets
    iface        = ui.latest_speed
    scores       = ui.latest_scores
    bf           = ui.band_filter

    cfg = get_mapping(bf)
    map_ch    = cfg["map_ch"]
    x_min     = cfg["x_min"]; x_max = cfg["x_max"]
    ticks_x   = cfg["ticks_x"]; ticks_label = cfg["ticks_label"]
    scale24   = cfg["sigma_scale_24"]; scale5 = cfg["sigma_scale_5"]

    # çizim eksenlerini temizle
    ui.ax.clear()
    ui.ax_info.clear()
    ui.ax_info.axis("off")

    # başlık ve eksen
    band_title = {"all":"Tümü","2.4":"2.4 GHz","5":"5 GHz"}[bf]
    ui.ax.set_title(
        f"Wi-Fi Kanal Spektrumu • Filtre: {band_title}",
        fontsize=13, fontweight="bold"
    )
    ui.ax.set_ylabel("dBm (sinyal gücü)")
    ui.ax.set_ylim(-100, -25)
    ui.ax.set_xlim(x_min, x_max)
    ui.ax.grid(True, linestyle="--", alpha=0.35)
    ui.ax.set_xticks(ticks_x)
    ui.ax.set_xticklabels(ticks_label, fontsize=8)

    if bf == "2.4":
        ui.ax.set_xlabel("Kanal (2.4 GHz)")
    elif bf == "5":
        ui.ax.set_xlabel("Kanal (5 GHz)")
    else:
        ui.ax.set_xlabel("Kanal (2.4 / 5 GHz)")

    # AP listesi hazırla
    aps = flatten_aps(nets_by_ssid, bf)

    if not aps:
        ui.ax.text(0.5,0.5,"Ağ bulunamadı / Wi-Fi kapalı olabilir",ha="center",va="center",
                   transform=ui.ax.transAxes,fontsize=11,color="gray")
    else:
        aps_sorted = sorted(aps, key=lambda a:a["dbm"], reverse=True)

        label_bins = defaultdict(int)
        x_line = np.linspace(x_min, x_max, 2000)
        base_floor = -100.0

        for ap in aps_sorted:
            ssid = ap["ssid"]
            ch   = ap["channel"]
            dbm  = ap["dbm"]
            band = ap["band"]
            x_center = map_ch(ch, band)
            if x_center is None:
                continue

            # görsel sigma
            base_sigma_24 = 1.6
            base_sigma_5  = 2.5
            if band == "2.4 GHz":
                sigma_vis = base_sigma_24 * (scale24 if scale24 else 1.0)
            else:
                sigma_vis = base_sigma_5  * (scale5  if scale5  else 1.0)

            y_curve = base_floor + gaussian_curve(x_line, x_center, sigma_vis, dbm)
            col = color_for_ssid(ssid)

            ui.ax.fill_between(
                x_line, base_floor, y_curve,
                color=col, alpha=0.30, linewidth=0
            )
            ui.ax.plot(
                x_line, y_curve,
                color=col, linewidth=1.8, alpha=0.95
            )

            # Etiket konumlandırma
            # Aynı görsel alanda bin_key ile gruplayıp üst üste binmeyi açıyoruz.
            bin_key = round(x_center)
            idx = label_bins[bin_key]
            label_bins[bin_key] += 1

            side = -1 if (idx % 2)==0 else 1      # sağ/sol zigzag
            step = (idx // 2) + 1
            x_jitter = side * 0.6 * step          # x sapması
            y_jitter = 4.0 + (idx * 3.5)          # yukarı kay

            label_x = x_center + x_jitter
            label_y = dbm + y_jitter

            ui.ax.text(
                label_x,
                label_y,
                ssid,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=col,
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="none",
                    boxstyle="round,pad=0.25"
                )
            )

    # Sağ panel bilgisi yaz
    side_lines = []
    # Bağlı ağ
    if iface.get("ssid"):
        side_lines.append(f"🔗 Bağlı: {iface['ssid']}")
        side_lines.append(f"Hız: ↓ {iface.get('rx','?')} Mbps  ↑ {iface.get('tx','?')} Mbps")
    else:
        side_lines.append("🔗 Bağlı ağ yok")

    # Ağ listesi
    side_lines.append("")
    side_lines.append(f"📶 Ağlar ({band_title})")
    for row in best_per_ssid(nets_by_ssid, bf):
        side_lines.append(
            f"• {row['ssid']} [{row['band']}, Ch:{row['channel']}] "
            f"{row['dbm']:.0f} dBm {stars_from_dbm(row['dbm'])}"
        )

    # En iyi kanallar
    if scores:
        side_lines.append("")
        side_lines.append("⭐ En Uygun Kanallar (düşük skor = daha boş):")
        for line in best_channels_text(scores, top_n=3):
            side_lines.append("  " + line)

    # Yenileme / Tuş bilgisi
    side_lines.append("")
    side_lines.append(
        f"⏱ Tarama: {SCAN_INTERVAL:.0f}s   Görsel: {PLOT_INTERVAL:.0f}s"
    )
    side_lines.append("Tuşlar: 1=2.4  2=5  3=Tümü  q=Çık")

    ui.ax_info.text(
        0.0, 1.0,
        "\n".join(side_lines),
        ha="left", va="top",
        fontsize=9.5,
        bbox=dict(
            facecolor="white",
            alpha=0.95,
            edgecolor="#999",
            boxstyle="round,pad=0.45"
        )
    )

    # Üst başlık
    ui.fig.suptitle(
        "Wi-Fi Analyzer Pro • Raspberry Edition — 1=2.4  2=5  3=Tümü  q=Çık",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0,0.08,1,0.92])

# ================== UI Sınıfı ==================
class WiFiAnalyzerUI:
    def __init__(self):
        self.band_filter = DEFAULT_FILTER
        self.auto_refresh = True
        self.running = True

        # canlı durum çerçevesi
        self.latest_nets   = {}
        self.latest_speed  = {}
        self.latest_scores = {}

        plt.ion()
        self.fig = plt.figure(figsize=(15,7))
        gs = self.fig.add_gridspec(6,8, wspace=0.5, hspace=0.6)
        self.ax      = self.fig.add_subplot(gs[0:6,0:5])
        self.ax_info = self.fig.add_subplot(gs[0:6,5:8])
        self.ax_info.axis("off")

        # Yenile butonu
        ax_btn = plt.axes([0.60, 0.06, 0.09, 0.07])
        self.btn = Button(ax_btn, 'Yenile')
        self.btn.on_clicked(lambda e: draw_plot(self))

        # Band seçimi
        ax_radio = plt.axes([0.72, 0.06, 0.12, 0.12])
        self.radio = RadioButtons(ax_radio, ('all','2.4','5'),
                                  active={'all':0,'2.4':1,'5':2}[self.band_filter])
        self.radio.on_clicked(self.on_radio)

        # Auto checkbox
        ax_check = plt.axes([0.86, 0.06, 0.08, 0.10])
        self.chk = CheckButtons(ax_check, ['Auto'], [self.auto_refresh])
        self.chk.on_clicked(self.on_check)

        # Klavye
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_radio(self, label):
        self.band_filter = label
        draw_plot(self)

    def on_check(self, label):
        self.auto_refresh = not self.auto_refresh

    def on_key(self, event):
        k=(event.key or "").lower()
        if k=='1':
            self.band_filter="2.4"
            self.radio.set_active(1)
            draw_plot(self)
        elif k=='2':
            self.band_filter="5"
            self.radio.set_active(2)
            draw_plot(self)
        elif k=='3':
            self.band_filter="all"
            self.radio.set_active(0)
            draw_plot(self)
        elif k=='q':
            self.running = False

# ================== Döngü ==================
def main_loop(ui):
    last_scan = 0.0
    last_plot = 0.0
    while ui.running:
        now = time.time()

        # 1) Tarama (arka plan) her SCAN_INTERVAL sn
        if now - last_scan >= SCAN_INTERVAL:
            nets_now  = scan_networks_raw()
            speed_now = scan_interface_speed_raw()
            scores_now = compute_channel_scores(nets_now)

            ui.latest_nets   = nets_now
            ui.latest_speed  = speed_now
            ui.latest_scores = scores_now

            last_scan = now

        # 2) Çizim sadece PLOT_INTERVAL sn'de bir
        if now - last_plot >= PLOT_INTERVAL:
            draw_plot(ui)
            last_plot = now

        # Tkinter/matplotlib event loop çalışsın
        plt.pause(0.05)

    plt.close(ui.fig)

# ================== Çalıştır ==================
if __name__ == "__main__":
    print("Wi-Fi Analyzer Pro • Raspberry Edition başlatılıyor…")
    ui = WiFiAnalyzerUI()
    try:
        main_loop(ui)
    except KeyboardInterrupt:
        print("\nÇıkış yapıldı.")
        sys.exit(0)


