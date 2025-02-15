import re
import math
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.gridspec as gridspec

# -------------------------------------
# 從 header 中解析實際震央 (假設 header 包含 "Lon:120.57" 與 "Lat:23.23")
# -------------------------------------
def parse_actual_epicenter(header):
    lon_match = re.search(r"Lon:([\d\.]+)", header)
    lat_match = re.search(r"Lat:([\d\.]+)", header)
    if lon_match and lat_match:
        return float(lon_match.group(1)), float(lat_match.group(1))
    else:
        return None

# ------------------------------
# 1. Haversine公式：計算兩點球面距離（單位：km）
# ------------------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # km
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ------------------------------
# 2. 解析測站資料檔 (2025007.txt)
# ------------------------------
def parse_station_data(filename):
    try:
        with open(filename, "r", encoding="big5") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filename, "r", encoding="cp950") as f:
            text = f.read()
    
    parts = text.split("Stacode=")
    header = parts[0].strip()  # 真實震源數據
    stations = []
    for part in parts[1:]:
        seg = "Stacode=" + part
        fields = seg.split(",")
        station = {}
        for field in fields:
            field = field.strip()
            if "=" in field:
                key, value = field.split("=", 1)
                station[key.strip()] = value.strip()
        required = ['Stacode', 'Stalon', 'Stalat', 'Int', 'PGA(SUM)', 'PGV(SUM)']
        if all(k in station for k in required):
            try:
                station['Stalon'] = float(station['Stalon'])
                station['Stalat'] = float(station['Stalat'])
                station['PGA(SUM)'] = float(station['PGA(SUM)'])
                station['PGV(SUM)'] = float(station['PGV(SUM)'])
            except Exception as e:
                print("數值轉換錯誤:", e)
                continue
            stations.append(station)
    return header, stations

# ------------------------------
# 3. 震度預估公式
# ------------------------------
def predict_pga(sta_lon, sta_lat, epi_lon, epi_lat, depth, mag):
    horizontal_dist = haversine(epi_lon, epi_lat, sta_lon, sta_lat)
    dist = math.sqrt(horizontal_dist**2 + depth**2)
    return 1.657 * math.exp(1.533 * mag) * (dist ** -1.607)

def predict_pgv(sta_lon, sta_lat, epi_lon, epi_lat, depth, mag):
    horizontal_dist = haversine(epi_lon, epi_lat, sta_lon, sta_lat)
    long_term = (10 ** (0.5 * mag - 1.85)) / 2.0
    hypo_dist = math.sqrt(depth**2 + horizontal_dist**2) - long_term
    x = max(hypo_dist, 3)
    term = x + 0.0028 * (10 ** (0.5 * mag))
    gpv600 = 10 ** (0.58 * mag + 0.0038 * depth - 1.29 - math.log10(term) - 0.002 * x)
    return gpv600 * 1.31

def get_predicted_intensity(pga, pgv):
    if pga < 0.8:
        return "0"
    elif pga < 2.5:
        return "1"
    elif pga < 8:
        return "2"
    elif pga < 25:
        return "3"
    elif pga < 80:
        return "4"
    else:
        if pgv < 15:
            return "4"
        elif 15 <= pgv < 30:
            return "5-"
        elif 30 <= pgv < 50:
            return "5+"
        elif 50 <= pgv < 80:
            return "6-"
        elif 80 <= pgv < 140:
            return "6+"
        elif pgv >= 140:
            return "7"
        else:
            return "?"

# ------------------------------
# 4. 震度文字與顏色設定
# ------------------------------
obs_intensity_map = {
    "7級": ("7", "red"),
    "6強": ("6+", "red"),
    "6弱": ("6-", "red"),
    "5強": ("5+", "red"),
    "5弱": ("5-", "red"),
    "4級": ("4", "green"),
    "3級": ("3", "green"),
    "2級": ("2", "blue"),
    "1級": ("1", "black")
}
pred_intensity_color = {
    "0": "black",
    "1": "black",
    "2": "blue",
    "3": "green",
    "4": "green",
    "5-": "red",
    "5+": "red",
    "6-": "red",
    "6+": "red",
    "7": "red",
    "?": "orange"
}

# ------------------------------
# 5. 主程式：合成圖像與存檔 (dpi=500)
# ------------------------------
def main():
    header, stations = parse_station_data("2025007.txt")
    print(f"Parsed {len(stations)} stations.")
    
    actual_epicenter = parse_actual_epicenter(header)
    
    epi_lon, epi_lat = 120.53, 23.28
    depth = 10.0
    mag = 6.4
    
    for sta in stations:
        lon, lat = sta["Stalon"], sta["Stalat"]
        pga = predict_pga(lon, lat, epi_lon, epi_lat, depth, mag)
        pgv = predict_pgv(lon, lat, epi_lon, epi_lat, depth, mag)
        sta["predicted_pga"] = pga
        sta["predicted_pgv"] = pgv
        sta["predicted_intensity"] = get_predicted_intensity(pga, pgv)
    
    counties = gpd.read_file("COUNTY_MOI_1090820.json")
    xmin, xmax, ymin, ymax = 119, 123, 21, 26
    
    # 計算誤差百分比： (Observed - Predicted) / Predicted * 100
    error_pga = [((obs - pred) / pred) * 100 for pred, obs in zip(
                  [sta["predicted_pga"] for sta in stations],
                  [sta["PGA(SUM)"] for sta in stations])]
    error_pgv = [((obs - pred) / pred) * 100 for pred, obs in zip(
                  [sta["predicted_pgv"] for sta in stations],
                  [sta["PGV(SUM)"] for sta in stations])]
    
    pga_pred = [sta["predicted_pga"] for sta in stations]
    pga_obs  = [sta["PGA(SUM)"] for sta in stations]
    pgv_pred = [sta["predicted_pgv"] for sta in stations]
    pgv_obs  = [sta["PGV(SUM)"] for sta in stations]
    pga_min_full = min(min(pga_pred), min(pga_obs))
    pga_max_full = max(max(pga_pred), max(pga_obs))
    pgv_min_full = min(min(pgv_pred), min(pgv_obs))
    pgv_max_full = max(max(pgv_pred), max(pgv_obs))
    
    # 建立大圖：左側2張地圖；右側6散佈圖 (左欄：PGA, 右欄：PGV)
    fig = plt.figure(figsize=(20,15))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.1)
    
    # 左側區域：2行 (高度比例 [1.5, 1.5])
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.2, height_ratios=[1.5, 1.5])
    ax_obs = fig.add_subplot(left_gs[0])
    ax_pred = fig.add_subplot(left_gs[1])
    
    # Observed Intensity Map
    counties.plot(ax=ax_obs, edgecolor="black", facecolor="white")
    ax_obs.set_title("Observed Intensity Map", fontsize=14)
    ax_obs.set_xlabel("Longitude", fontsize=12)
    ax_obs.set_ylabel("Latitude", fontsize=12)
    ax_obs.set_xlim(xmin, xmax)
    ax_obs.set_ylim(ymin, ymax)
    for sta in stations:
        lon, lat = sta["Stalon"], sta["Stalat"]
        if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
            continue
        obs_int = sta["Int"]
        label, color = obs_intensity_map.get(obs_int, (obs_int, "black"))
        ax_obs.text(lon, lat, label, color=color, fontsize=12, ha="center", va="center")
    if actual_epicenter is not None:
        ax_obs.plot(actual_epicenter[0], actual_epicenter[1], marker="*", markersize=12,
                    color="magenta", markeredgecolor="black", markeredgewidth=1,
                    linestyle="None", label="Actual Epicenter")
    ax_obs.legend(loc="upper right", fontsize=10)
    ax_obs.text(0.02, 0.02, header, transform=ax_obs.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.7))
    ax_obs.set_box_aspect(1)
    
    # Predicted Intensity Map
    counties.plot(ax=ax_pred, edgecolor="black", facecolor="white")
    ax_pred.set_title("Predicted Intensity Map", fontsize=14)
    ax_pred.set_xlabel("Longitude", fontsize=12)
    ax_pred.set_ylabel("Latitude", fontsize=12)
    ax_pred.set_xlim(xmin, xmax)
    ax_pred.set_ylim(ymin, ymax)
    for sta in stations:
        lon, lat = sta["Stalon"], sta["Stalat"]
        if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
            continue
        pred_int = sta["predicted_intensity"]
        color = pred_intensity_color.get(pred_int, "black")
        ax_pred.text(lon, lat, pred_int, color=color, fontsize=12, ha="center", va="center")
    if actual_epicenter is not None:
        ax_pred.plot(actual_epicenter[0], actual_epicenter[1], marker="*", markersize=12,
                     color="magenta", markeredgecolor="black", markeredgewidth=1,
                     linestyle="None", label="Actual Epicenter")
    ax_pred.plot(epi_lon, epi_lat, marker="X", markersize=10,
                 color="orange", markeredgecolor="black", markeredgewidth=1,
                 linestyle="None", label="Predicted Epicenter")
    ax_pred.legend(loc="upper right", fontsize=10)
    ax_pred.text(0.02, 0.02, f"Predicted Epicenter:\nLon: {epi_lon}\nLat: {epi_lat}\nDepth: {depth} km\nMag: {mag}",
                 transform=ax_pred.transAxes, fontsize=9,
                 verticalalignment="bottom", horizontalalignment="left",
                 bbox=dict(facecolor="white", alpha=0.7))
    ax_pred.set_box_aspect(1)
    
    # 右側區域：3 rows x 2 cols (左欄: PGA, 右欄: PGV)
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[1], hspace=0.2, wspace=0.1)
    ax_pga_full = fig.add_subplot(right_gs[0, 0])
    ax_pga_500  = fig.add_subplot(right_gs[1, 0])
    ax_pga_200  = fig.add_subplot(right_gs[2, 0])
    ax_pgv_full = fig.add_subplot(right_gs[0, 1])
    ax_pgv_30   = fig.add_subplot(right_gs[1, 1])
    ax_pgv_15   = fig.add_subplot(right_gs[2, 1])
    
    # 使用 error 數據著色 (error = ((Observed - Predicted)/Predicted)*100)
    sc1 = ax_pga_full.scatter(pga_pred, pga_obs, c=error_pga, cmap="bwr", vmin=-100, vmax=100, 
                               edgecolors="k", alpha=0.7)
    ax_pga_full.plot([pga_min_full, pga_max_full], [pga_min_full, pga_max_full], "r--")
    ax_pga_full.set_title("Predicted vs Observed PGA (Full)", fontsize=12)
    ax_pga_full.set_xlabel("Predicted PGA", fontsize=10)
    ax_pga_full.set_ylabel("Observed PGA", fontsize=10)
    ax_pga_full.set_xlim(pga_min_full, pga_max_full)
    ax_pga_full.set_ylim(pga_min_full, pga_max_full)
    ax_pga_full.set_box_aspect(1)
    
    sc2 = ax_pga_500.scatter(pga_pred, pga_obs, c=error_pga, cmap="bwr", vmin=-100, vmax=100, 
                               edgecolors="k", alpha=0.7)
    ax_pga_500.plot([0, 500], [0, 500], "r--")
    ax_pga_500.set_title("PGA (<=500 gal)", fontsize=12)
    ax_pga_500.set_xlabel("Predicted PGA", fontsize=10)
    ax_pga_500.set_ylabel("Observed PGA", fontsize=10)
    ax_pga_500.set_xlim(0, 500)
    ax_pga_500.set_ylim(0, 500)
    ax_pga_500.set_box_aspect(1)
    
    sc3 = ax_pga_200.scatter(pga_pred, pga_obs, c=error_pga, cmap="bwr", vmin=-100, vmax=100, 
                               edgecolors="k", alpha=0.7)
    ax_pga_200.plot([0, 200], [0, 200], "r--")
    ax_pga_200.set_title("PGA (<=200 gal)", fontsize=12)
    ax_pga_200.set_xlabel("Predicted PGA", fontsize=10)
    ax_pga_200.set_ylabel("Observed PGA", fontsize=10)
    ax_pga_200.set_xlim(0, 200)
    ax_pga_200.set_ylim(0, 200)
    ax_pga_200.set_box_aspect(1)
    
    sc4 = ax_pgv_full.scatter(pgv_pred, pgv_obs, c=error_pgv, cmap="bwr", vmin=-100, vmax=100, 
                               edgecolors="k", alpha=0.7)
    ax_pgv_full.plot([pgv_min_full, pgv_max_full], [pgv_min_full, pgv_max_full], "r--")
    ax_pgv_full.set_title("Predicted vs Observed PGV (Full)", fontsize=12)
    ax_pgv_full.set_xlabel("Predicted PGV", fontsize=10)
    ax_pgv_full.set_ylabel("Observed PGV", fontsize=10)
    ax_pgv_full.set_xlim(pgv_min_full, pgv_max_full)
    ax_pgv_full.set_ylim(pgv_min_full, pgv_max_full)
    ax_pgv_full.set_box_aspect(1)
    
    sc5 = ax_pgv_30.scatter(pgv_pred, pgv_obs, c=error_pgv, cmap="bwr", vmin=-100, vmax=100, 
                             edgecolors="k", alpha=0.7)
    ax_pgv_30.plot([0, 30], [0, 30], "r--")
    ax_pgv_30.set_title("PGV (<=30 kine)", fontsize=12)
    ax_pgv_30.set_xlabel("Predicted PGV", fontsize=10)
    ax_pgv_30.set_ylabel("Observed PGV", fontsize=10)
    ax_pgv_30.set_xlim(0, 30)
    ax_pgv_30.set_ylim(0, 30)
    ax_pgv_30.set_box_aspect(1)
    
    sc6 = ax_pgv_15.scatter(pgv_pred, pgv_obs, c=error_pgv, cmap="bwr", vmin=-100, vmax=100, 
                             edgecolors="k", alpha=0.7)
    ax_pgv_15.plot([0, 15], [0, 15], "r--")
    ax_pgv_15.set_title("PGV (<=15 kine)", fontsize=12)
    ax_pgv_15.set_xlabel("Predicted PGV", fontsize=10)
    ax_pgv_15.set_ylabel("Observed PGV", fontsize=10)
    ax_pgv_15.set_xlim(0, 15)
    ax_pgv_15.set_ylim(0, 15)
    ax_pgv_15.set_box_aspect(1)
    
    # 新增共用 colorbar, 高度與右側區域相同
    cax = fig.add_axes([0.942, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(sc1, cax=cax, orientation="vertical")
    cbar.set_label("Prediction Error (%)", fontsize=12)
    
    # 新增總大標題 (翻譯: "No.114007 TREM-EEW Seismic Intensity Prediction Performance")
    fig.suptitle("No.114007 TREM-EEW Intensity Prediction Performance", fontsize=20)
    
    fig.subplots_adjust(left=0.03, right=0.92, top=0.93, bottom=0.03)
    fig.savefig("Composite_All.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    main()
