---
title: "Cycle diurne de la température d'été à Lyon-Bron, par décennie"
layout: post
comments: true
author: François Pacull
date: 2026-07-04
categories: [Python, Climat]
tags:
- Python
- pandas
- Matplotlib
- SciPy
- Météo-France
- open data
- température
- Lyon
image: /img/2026-07-04_01/output_11_0.png
---

Météo-France publie ses [données climatologiques de base horaires](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-horaires/) en open data sur data.gouv.fr. On les utilise ici pour tracer la température moyenne d'été (juin–juillet–août) à chaque heure de la journée à la station Lyon-Bron (`NUM_POSTE` 69029001), une courbe par décennie de 1971 à 2025, en heure locale d'été (CEST, UTC+2). Le notebook est autonome et reproductible : les données sont téléchargées directement via l'API data.gouv.fr puis mises en cache localement, aucun fichier local n'est requis.

La fréquence d'échantillonnage change sur la période :
- **avant 1991** : ~8 relevés/jour aux heures synoptiques 3-horaires (0,3,…,21 UTC → 2,5,…,23 en heure locale).
  Ces décennies n'ont que 8 points/jour : on les rend en pointillé, points relevés reliés par une
  spline cubique périodique ([`CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) de [SciPy](https://scipy.org/)) ;
- **à partir de 1991** : vraies données horaires (24 points) → trait plein entre les points.

La palette `rainbow_PuRd` de [Paul Tol](https://sronpersonalpages.nl/~pault/) (paquet [tol-colors](https://github.com/Descanonge/tol_colors)) ordonne les décennies des teintes froides vers les teintes chaudes et reste lisible pour tous les types de daltonisme.

## 1. Installation des dépendances

À exécuter une seule fois (versions figées pour la reproductibilité).


```python
# %pip install "pandas==3.0.3" "numpy==2.4.6" "matplotlib==3.11.0" "scipy==1.18.0" "tol-colors==2.2.0"
```


```python
import io
import json
import re
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tol_colors as tc
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
```

## 2. Télécharger la température horaire de Lyon-Bron

On interroge l'API data.gouv.fr du jeu *« Données climatologiques de base – horaires »*, on ne garde que
les fichiers du département 69 (Rhône) couvrant 1971+, et on en extrait la station Lyon-Bron
(`NUM_POSTE = 69029001`) : deux colonnes suffisent, l'horodatage (`AAAAMMJJHH`, UTC) et la température
`T` (°C). Le résultat est mis en cache localement (`lyon_bron_hourly.parquet`) pour que les ré-exécutions
soient instantanées.


```python
DATASET_API = (
    "https://www.data.gouv.fr/api/1/datasets/"
    "donnees-climatologiques-de-base-horaires/"
)
STATION_ID = "69029001"  # Lyon-Bron
STATION_NAME = "Lyon-Bron"
FIRST_YEAR = 1971  # 1re décennie tracée
LAST_FULL_YEAR = 2025  # 2026 partielle exclue
CACHE = Path("lyon_bron_hourly.parquet")
UA = {"User-Agent": "Mozilla/5.0 (lyon-diurnal-notebook)"}


def hourly_file_urls():
    """URLs des fichiers horaires dept-69 dont la période atteint FIRST_YEAR."""
    req = urllib.request.Request(DATASET_API, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.load(resp)
    urls = [
        r["url"]
        for r in meta.get("resources", [])
        if "H_69_" in (r.get("url") or "") and (r.get("url") or "").endswith(".csv.gz")
    ]
    keep = []
    for u in urls:
        years = [int(y) for y in re.findall(r"(\d{4})", u.rsplit("/", 1)[-1])]
        if years and max(years) >= FIRST_YEAR:  # ex. H_69_1970-1979 -> 1979 >= 1971
            keep.append(u)
    return sorted(keep)


def load_lyon_bron():
    """DataFrame [datetime (UTC), T (°C)] pour Lyon-Bron, depuis l'open data (avec cache)."""
    if CACHE.exists():
        print(f"Cache trouvé : {CACHE}")
        return pd.read_parquet(CACHE)

    frames = []
    for url in hourly_file_urls():
        name = url.rsplit("/", 1)[-1]
        print(f"téléchargement {name} …", flush=True)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as resp:
            blob = resp.read()
        df = pd.read_csv(
            io.BytesIO(blob),
            sep=";",
            compression="gzip",
            usecols=["NUM_POSTE", "AAAAMMJJHH", "T"],
            dtype={"NUM_POSTE": "string", "AAAAMMJJHH": "string", "T": "string"},
        )
        df = df[df["NUM_POSTE"] == STATION_ID]
        if not df.empty:
            frames.append(df[["AAAAMMJJHH", "T"]])

    raw = pd.concat(frames, ignore_index=True)
    out = (
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(raw["AAAAMMJJHH"], format="%Y%m%d%H"),
                "T": pd.to_numeric(raw["T"], errors="coerce"),
            }
        )
        .dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    out.to_parquet(CACHE)
    print(f"{len(out):,} lignes mises en cache -> {CACHE}")
    return out


df = load_lyon_bron()
print(
    f"{len(df):,} lignes | {df['datetime'].min():%Y-%m-%d} \u2192 {df['datetime'].max():%Y-%m-%d}"
)
df.head()
```

    Cache trouvé : lyon_bron_hourly.parquet
    374,127 lignes | 1970-01-01 → 2026-07-04





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01 00:00:00</td>
      <td>-2.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-01-01 03:00:00</td>
      <td>-2.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-01-01 06:00:00</td>
      <td>-2.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-01-01 09:00:00</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-01-01 12:00:00</td>
      <td>-1.8</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Découpage en décennies

De 1971 à la dernière année complète. La dernière décennie peut être partielle (ex. 2021-2025).


```python
def decade_bounds(first=FIRST_YEAR, last=LAST_FULL_YEAR):
    """Liste de (début, fin) par tranche de 10 ans."""
    out, start = [], first
    while start <= last:
        out.append((start, min(start + 9, last)))
        start += 10
    return out


decades = decade_bounds()
decades
```




    [(1971, 1980),
     (1981, 1990),
     (1991, 2000),
     (2001, 2010),
     (2011, 2020),
     (2021, 2025)]



## 4. Profil diurne moyen par (décennie, heure locale)

On filtre l'été (JJA), on convertit en heure locale (`(heure_UTC + 2) % 24`), puis on moyenne la
température pour chaque couple (décennie, heure).


```python
lo = decades[0][0]
h = (
    df[
        (df["datetime"].dt.year >= lo)
        & (df["datetime"].dt.year <= LAST_FULL_YEAR)
        & (df["datetime"].dt.month.isin([6, 7, 8]))
    ]
    .dropna(subset=["T"])
    .copy()
)

# Heure locale d'été à Lyon = UTC+2. Les relevés 3-horaires tombent aux heures 2,5,8,11,14,17,20,23.
h["hour"] = (h["datetime"].dt.hour + 2) % 24

bins = [decades[0][0] - 1] + [e for _, e in decades]
labels = [f"{a}\u2013{b}" for a, b in decades]
h["period"] = pd.cut(h["datetime"].dt.year, bins=bins, labels=labels)

prof = h.groupby(["period", "hour"], observed=True)["T"].mean().unstack("period")
prof.round(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>period</th>
      <th>1971–1980</th>
      <th>1981–1990</th>
      <th>1991–2000</th>
      <th>2001–2010</th>
      <th>2011–2020</th>
      <th>2021–2025</th>
    </tr>
    <tr>
      <th>hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>19.5</td>
      <td>20.3</td>
      <td>20.6</td>
      <td>20.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>18.8</td>
      <td>19.7</td>
      <td>20.0</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.4</td>
      <td>17.4</td>
      <td>18.2</td>
      <td>19.0</td>
      <td>19.3</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>17.5</td>
      <td>18.4</td>
      <td>18.7</td>
      <td>18.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>17.0</td>
      <td>17.9</td>
      <td>18.2</td>
      <td>18.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.9</td>
      <td>15.8</td>
      <td>16.5</td>
      <td>17.4</td>
      <td>17.6</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>16.1</td>
      <td>17.0</td>
      <td>17.2</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>16.2</td>
      <td>17.0</td>
      <td>17.2</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15.7</td>
      <td>16.5</td>
      <td>17.4</td>
      <td>18.2</td>
      <td>18.6</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>18.7</td>
      <td>19.7</td>
      <td>20.1</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>20.1</td>
      <td>21.0</td>
      <td>21.5</td>
      <td>22.4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>19.8</td>
      <td>20.7</td>
      <td>21.4</td>
      <td>22.3</td>
      <td>22.8</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>22.7</td>
      <td>23.5</td>
      <td>24.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>23.7</td>
      <td>24.5</td>
      <td>25.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>22.9</td>
      <td>23.8</td>
      <td>24.5</td>
      <td>25.2</td>
      <td>25.9</td>
      <td>27.1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>25.1</td>
      <td>25.8</td>
      <td>26.5</td>
      <td>27.7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>25.4</td>
      <td>26.1</td>
      <td>26.9</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>23.9</td>
      <td>24.9</td>
      <td>25.5</td>
      <td>26.3</td>
      <td>26.9</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>25.2</td>
      <td>26.0</td>
      <td>26.7</td>
      <td>27.7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>24.6</td>
      <td>25.5</td>
      <td>26.1</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22.2</td>
      <td>23.2</td>
      <td>23.7</td>
      <td>24.6</td>
      <td>25.2</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>22.4</td>
      <td>23.4</td>
      <td>23.8</td>
      <td>24.5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>21.1</td>
      <td>22.1</td>
      <td>22.4</td>
      <td>22.8</td>
    </tr>
    <tr>
      <th>23</th>
      <td>18.5</td>
      <td>19.6</td>
      <td>20.3</td>
      <td>21.1</td>
      <td>21.4</td>
      <td>21.7</td>
    </tr>
  </tbody>
</table>
</div>



Avant 1991, seules les 8 heures synoptiques locales (2, 5, 8, 11, 14, 17, 20, 23) ont une valeur, d'où les `<NA>` ; 17 h, l'heure du pic, fait partie des heures réellement observées sur toute la période.

## 5. Figure

- décennies **3-horaires** (≤ 8 heures disponibles) : points relevés + spline cubique périodique (pointillé) ;
- décennies **horaires** : trait plein, bouclé (T à 24 h = T à 0 h).


```python
cmap = tc.rainbow_PuRd
fig, ax = plt.subplots(figsize=(11, 6.0))
handles, leg_labels = [], []

n = len(labels)
for i, period in enumerate(labels):
    if period not in prof.columns:
        continue
    color = cmap(i / (n - 1))
    col = prof[period].dropna()

    if len(col) <= 8:
        # Décennie 3-horaire : spline cubique périodique sur les 8 points synoptiques.
        xh = col.index.to_numpy(dtype=float)  # heures locales, ex. 2..23
        yh = col.to_numpy(dtype=float)
        xp = np.append(xh, xh[0] + 24.0)  # une période de 24 h
        yp = np.append(yh, yh[0])
        cs = CubicSpline(xp, yp, bc_type="periodic")  # C2 + raccord continu à la couture 2 h/26 h
        xx = np.linspace(0.0, 24.0, 289)
        xq = np.where(xx < xh[0], xx + 24.0, xx)  # replie les heures avant le 1er point
        ax.plot(xx, cs(xq), color=color, lw=2.2, ls=":")
        ax.plot(xh, yh, ls="none", marker="o", ms=5, color=color)
        handles.append(Line2D([], [], color=color, lw=2.2, ls=":", marker="o", ms=5))
        leg_labels.append(f"{period} (3-h, interpolé)")
    else:
        # Décennie horaire : trait plein, bouclé.
        xx = np.append(col.index.to_numpy(dtype=float), 24.0)
        yy = np.append(col.to_numpy(dtype=float), col.loc[0])
        (line,) = ax.plot(xx, yy, marker="o", ms=4, lw=2.2, color=color)
        handles.append(line)
        leg_labels.append(period)

ax.set_xlabel("Heure locale (CEST, UTC+2)")
ax.set_ylabel("Température moyenne (°C)")
ax.set_xticks(range(0, 25, 3))
ax.set_xlim(-0.4, 24.4)
ax.set_title(
    f"Cycle diurne moyen de la température en été (juin-juillet-août) par décennie à {STATION_NAME}",
    fontsize=13,
    color="0.25",
    pad=12,
)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(color="0.92", lw=0.8)
ax.legend(
    handles,
    leg_labels,
    loc="upper left",
    frameon=False,
    fontsize=9,
    title="Décennie",
)

fig.tight_layout()
fig.savefig("lyon_diurnal_cycle_by_decade.png", dpi=150)
```


    
<p align="center">
  <img width="900" src="/img/2026-07-04_01/output_11_0.png" alt="Cycle diurne moyen de la température en été (juin-juillet-août) par décennie à Lyon-Bron">
</p>

L'écart entre décennies est présent à toute heure et maximal en fin d'après-midi : au pic de 17 h, la température moyenne passe de 23,9 °C (1971–1980) à 28,0 °C (2021–2025).

La série est brute et la dernière tranche ne couvre que 5 ans ; la décennie complète 2016–2025 donne 27,9 °C au même pic. La croissance urbaine autour de Bron contribue peut-être un peu.
