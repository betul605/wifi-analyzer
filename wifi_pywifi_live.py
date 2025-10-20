# wifi_pywifi_live.py
import time
import pywifi
from pywifi import const
import matplotlib.pyplot as plt

def scan_with_pywifi():
    wifi = pywifi.PyWiFi()
    ifaces = wifi.interfaces()
    if not ifaces:
        return {}
    iface = ifaces[0]
    iface.scan()
    time.sleep(1.2)  # tarama için kısa bekle
    results = iface.scan_results()
    nets = {}
    for r in results:
        ssid = r.ssid or "Gizli Ağ"
        signal = max(0, min(100, int((r.signal + 100) * 2)))  # r.signal genelde dBm, dönüştürme
        dbm = r.signal
        # bazı kartlar channel/band vermez; pywifi'da channel yok olabilir
        ch = getattr(r, 'freq', None) or "-"
        nets.setdefault(ssid, []).append((signal, dbm, ch))
    return nets

def plot_networks(networks):
    plt.clf()
    ssids, signals, labels = [], [], []
    for ssid, ap_list in networks.items():
        avg = sum(x[0] for x in ap_list)/len(ap_list)
        dbm = sum(x[1] for x in ap_list)/len(ap_list)
        ch = ap_list[0][2]
        ssids.append(ssid)
        signals.append(avg)
        labels.append(f"{avg:.0f}% | {dbm:.1f} dBm | Ch:{ch}")
    bars = plt.barh(ssids, signals, color="deepskyblue", edgecolor="black")
    plt.title("📶 pywifi Canlı Wi-Fi Analizörü")
    plt.xlabel("Sinyal Gücü (%)")
    plt.xlim(0, 100)
    for bar,label in zip(bars, labels):
        plt.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, label, va="center", fontsize=9)
    plt.tight_layout()
    plt.pause(0.1)

def live(refresh=2):
    plt.ion()
    while True:
        nets = scan_with_pywifi()
        if nets:
            plot_networks(nets)
        else:
            plt.clf(); plt.text(0.4,0.5,"Ağ bulunamadı", fontsize=12); plt.pause(0.1)
        time.sleep(refresh)

if __name__ == "__main__":
    print("pywifi ile tarama başlatılıyor...")
    try:
        live(refresh=2)
    except KeyboardInterrupt:
        print("Çıkılıyor...")
