# Football Predictor — instrukcja obsługi

System prognozowania wyników meczów piłkarskich (H / D / A) dla 6 lig europejskich.
Dane pochodzą z [football-data.co.uk](https://www.football-data.co.uk).

---

## Spis treści

1. [Co robi program](#1-co-robi-program)
2. [Wymagania i instalacja](#2-wymagania-i-instalacja)
3. [Szybki start](#3-szybki-start)
4. [Tryby pracy](#4-tryby-pracy)
5. [Komendy w terminalu (CLI)](#5-komendy-w-terminalu-cli)
6. [Aplikacja webowa (Streamlit)](#6-aplikacja-webowa-streamlit)
7. [Notebook Jupyter](#7-notebook-jupyter)
8. [Modele prognozujące](#8-modele-prognozujące)
9. [Pliki i foldery](#9-pliki-i-foldery)
10. [Typowy cykl pracy](#10-typowy-cykl-pracy)
11. [Jak czytać prognozy](#11-jak-czytać-prognozy)
12. [Rozwiązywanie problemów](#12-rozwiązywanie-problemów)
13. [Najczęstsze pytania](#13-najczęstsze-pytania)

---

## 1. Co robi program

Program:

1. **Pobiera dane** — wyniki meczów z ostatnich 5 sezonów, kursy bukmacherskie i statystyki (strzały, rogi itd.).
2. **Buduje bazę** — zapisuje mecze w lokalnej bazie SQLite.
3. **Trenuje modele** — Dixon-Coles (statystyczny) i XGBoost (uczenie maszynowe).
4. **Prognozuje** — przewiduje wynik najbliższej kolejki w każdej z 6 lig.
5. **Aktualizuje się** — po rozegraniu kolejki dopisuje wyniki i trenuje modele od nowa.

### Obsługiwane ligi

| Kod | Liga |
|-----|------|
| E0  | Premier League (Anglia) |
| SP1 | La Liga (Hiszpania) |
| D1  | Bundesliga (Niemcy) |
| I1  | Serie A (Włochy) |
| F1  | Ligue 1 (Francja) |
| N1  | Eredivisie (Holandia) |

### Co oznaczają wyniki prognoz

| Symbol | Znaczenie |
|--------|-----------|
| **H** | Wygrana gospodarzy (Home) |
| **D** | Remis (Draw) |
| **A** | Wygrana gości (Away) |

---

## 2. Wymagania i instalacja

### Wymagania

- **Python 3.10+** (zalecane 3.11 lub 3.12)
- Połączenie z internetem (pobieranie danych CSV)
- Ok. 200 MB wolnego miejsca na dysku (baza + modele + surowe pliki)

### Instalacja krok po kroku

Otwórz terminal (PowerShell lub wiersz poleceń) i wykonaj:

```powershell
cd C:\Data_Science\08_Machine_learning_1\football_predictor
pip install -r requirements.txt
```

Instalowane biblioteki: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `requests`, `joblib`, `streamlit`.

### Sprawdzenie, czy działa

```powershell
python main.py status
```

Jeśli baza istnieje, zobaczysz liczbę meczów. Przy pierwszym uruchomieniu baza może być pusta — wtedy wykonaj `init` (patrz sekcja 3).

---

## 3. Szybki start

### Pierwsze uruchomienie (jednorazowo)

```powershell
cd "TWOJA ŚCIEŻKA DOSTĘPU DO KATALOGU"
python main.py init
```

Trwa **kilka–kilkanaście minut**. Program:
- pobierze ok. 5 lat historii z 6 lig,
- zapisze dane do bazy,
- wytrenuje oba modele,
- porówna je ze sobą,
- wygeneruje prognozy na najbliższą kolejkę.

Wyniki prognoz: plik `output/predictions.csv`.

### Co robić co tydzień

```powershell
# 1. Po zakończeniu kolejki — dopisz wyniki i przelicz modele
python main.py update

# 2. Zobacz prognozy (jeśli update ich nie wygenerował)
python main.py predict
```

### Alternatywa: aplikacja webowa

```powershell
streamlit run app.py
```

Otworzy się przeglądarka (domyślnie `http://localhost:8501`).

---

## 4. Tryby pracy

Program można obsługiwać na trzy sposoby:

| Tryb | Plik | Dla kogo |
|------|------|----------|
| **Terminal (CLI)** | `main.py` | Szybkie komendy, automatyzacja, harmonogram zadań |
| **Aplikacja webowa** | `app.py` | Wygodny interfejs, wykresy, analiza drużyn |
| **Notebook Jupyter** | `prognozowanie_pilkarskie.ipynb` | Eksperymenty, nauka, modyfikacja krok po kroku |

Wszystkie tryby korzystają z **tej samej bazy danych** i **tych samych modeli**.

---

## 5. Komendy w terminalu (CLI)

Wszystkie komendy uruchamiasz z folderu `football_predictor`:

```powershell
cd "TWOJA ŚCIEŻKA DOSTĘPU DO KATALOGU"
python main.py <komenda>
```

### `init` — pierwsza konfiguracja

```powershell
python main.py init
```

| Co robi | Pobiera historię, trenuje modele, porównuje je, generuje prognozy |
| Kiedy używać | Tylko raz na początku lub gdy chcesz zresetować cały system |
| Czas | Kilka–kilkanaście minut |

---

### `predict` — generowanie prognoz

```powershell
python main.py predict
python main.py predict --model xgboost
python main.py predict --model dixon_coles
python main.py predict --model auto
```

| Co robi | Prognozuje wyniki najbliższej kolejki we wszystkich ligach |
| Kiedy używać | Gdy chcesz odświeżyć prognozy bez pełnej aktualizacji bazy |
| Wynik | Plik `output/predictions.csv` |

**Wybór modelu (`--model`):**

| Wartość | Opis |
|---------|------|
| `auto` | Domyślnie — wybiera lepszy model z ostatniego porównania |
| `xgboost` | Zawsze XGBoost |
| `dixon_coles` | Zawsze Dixon-Coles |

---

### `update` — aktualizacja po kolejce

```powershell
python main.py update
python main.py update --force
```

| Co robi | Pobiera nowe wyniki, dopisuje do bazy, trenuje modele, generuje prognozy |
| Kiedy używać | Po rozegraniu kolejki (np. w niedzielę wieczorem lub w poniedziałek) |

**Bez `--force`:** program czeka, aż **wszystkie** mecze z ostatniej prognozy będą miały wynik w bazie.

**Z `--force`:** aktualizacja wymuszona nawet przy niekompletnej kolejce (np. przełożone mecze).

---

### `refresh` — odświeżenie kursów i statystyk

```powershell
python main.py refresh
```

| Co robi | Ponownie pobiera pliki CSV z football-data.co.uk i uzupełnia kursy/statystyki |
| Kiedy używać | Gdy w bazie brakuje kursów lub statystyk strzałów |
| Czas | Ok. 30 sekund |

---

### `compare` — porównanie modeli

```powershell
python main.py compare
```

| Co robi | Trenuje oba modele na danych historycznych i porównuje dokładność |
| Kiedy używać | Gdy chcesz sprawdzić, który model jest lepszy |
| Czas | 1–3 minuty |
| Wynik | Raport w `output/model_comparison.txt` |

Podział danych: **80% trening / 20% test**, chronologicznie (starsze mecze → trening, nowsze → test).

---

### `status` — stan systemu

```powershell
python main.py status
```

Wyświetla m.in.:
- liczbę meczów w bazie,
- liczbę rozegranych meczów,
- czy istnieją aktywne prognozy,
- czy prognozowana kolejka została już rozegrana.

---

### `set-token` — token API (opcjonalnie)

```powershell
python main.py set-token TWOJ_TOKEN
```

Dotyczy tylko opcjonalnego źródła **football-data.org**. Domyślnie program używa **football-data.co.uk** i token **nie jest potrzebny**.

---

## 6. Aplikacja webowa (Streamlit)

### Uruchomienie

```powershell
cd C:\Data_Science\08_Machine_learning_1\football_predictor
streamlit run app.py
```

Przeglądarka otworzy się automatycznie. Aby zatrzymać serwer: `Ctrl+C` w terminalu.

### Panel boczny (sterowanie)

| Przycisk | Odpowiednik CLI | Opis |
|----------|-----------------|------|
| **Inicjalizacja bazy** | `init` (tylko pobieranie) | Pobiera 5 lat historii |
| **Generuj prognozy** | `predict` | Tworzy prognozy wybranym modelem |
| **Porównaj modele** | `compare` | Ewaluacja XGBoost vs Dixon-Coles |
| **Odśwież dane** | `refresh` | Ponowne pobranie CSV |
| **Aktualizuj po kolejce** | `update` | Dopisanie wyników + retrening |
| Checkbox *Wymuś aktualizację* | `--force` | Pomija czekanie na pełną kolejkę |

Na dole panelu: metryki bazy (liczba meczów, kursy, status kolejki).

### Zakładki

#### Prognozy
- Lista meczów najbliższej kolejki z prognozą H/D/A.
- Filtry: **liga**, **data od–do**.
- Wykres prawdopodobieństw H / Remis / A dla każdego meczu.
- Pasek pewności prognozy.

#### Analiza drużyn
- Wybór ligi i drużyny.
- **Wykres Elo** — siła drużyny w czasie.
- **Forma** — ostatnie 10 meczów (punkty, wynik W/D/L).
- Metryki: aktualne Elo, średnia forma, średnia bramek.

#### Historia prognoz
- Archiwum wszystkich wygenerowanych prognoz.
- Porównanie z rzeczywistymi wynikami (✓ trafione / ✗ nietrafione).
- Dokładność ogólna, wg modelu i wg ligi.
- Prognozy zapisują się automatycznie przy każdym **Generuj prognozy**.

#### Porównanie modeli
- Dokładność Dixon-Coles i XGBoost na zbiorze testowym.
- Pełny raport tekstowy.

#### Tabela prognoz
- Surowa tabela wszystkich prognoz.
- Przycisk **Pobierz CSV**.

#### Instrukcja
- Pełna instrukcja obsługi programu (ten dokument) wbudowana w aplikację.
- Przycisk **Pobierz instrukcję (Markdown)** — zapis pliku na dysk.

---

## 7. Notebook Jupyter

Plik: `prognozowanie_pilkarskie.ipynb`

### Uruchomienie

1. Otwórz folder `football_predictor` w Jupyter Lab / VS Code / Cursor.
2. Uruchamiaj komórki **od góry do dołu**.

### Zawartość notebooka (logiczny przebieg)

1. Import modułów i konfiguracja.
2. Pobranie / odświeżenie danych z football-data.co.uk.
3. Podgląd bazy danych.
4. Trening modeli (opcjonalnie — domyślnie wyłączone, żeby nie trenować dwa razy).
5. Prognozy najbliższej kolejki.
6. Porównanie Dixon-Coles vs XGBoost.

Notebook jest przeznaczony do **eksperymentów i nauki**. Do codziennej pracy wygodniejsze są CLI lub Streamlit.

---

## 8. Modele prognozujące

Program oferuje dwa modele. Oba przewidują wynik meczu jako **H**, **D** lub **A**.

### Dixon-Coles

- Klasyczny model statystyczny oparty na rozkładzie Poissona.
- Wykorzystuje: drużyny, liczbę bramek, datę, ligę.
- **Nie** korzysta z kursów bukmacherskich.
- Szybki trening (ok. 1–2 sekundy).
- Dobry punkt odniesienia (baseline).

### XGBoost

- Model uczenia maszynowego (gradient boosting).
- Dodatkowe cechy:
  - rating **Elo** drużyn,
  - **forma** z ostatnich 5 meczów (punkty, bramki),
  - **kursy** bukmacherskie (AvgH, AvgD, AvgA),
  - **statystyki strzałów** i rogów z ostatnich meczów.
- Trening trwa dłużej (kilkadziesiąt sekund do kilku minut).
- Zazwyczaj nieco lepszy lub porównywalny z Dixon-Coles.

### Tryb `auto`

Program wybiera model, który miał wyższą dokładność w ostatnim poleceniu `compare`.
Wynik porównania zapisywany jest w `output/model_comparison.txt`.

### Oczekiwana dokładność

Oba modele osiągają ok. **47–50%** trafności na zbiorze testowym.
Dla porównania: losowy typ daje ~33%, a faworyt bukmachera ~50–55%.
**Prognozy nie gwarantują wygranych** — to narzędzie analityczne, nie system bukmacherski.

---

## 9. Pliki i foldery

```
football_predictor/
├── main.py                          # Program główny (CLI)
├── app.py                           # Aplikacja Streamlit
├── prognozowanie_pilkarskie.ipynb   # Notebook Jupyter
├── INSTRUKCJA.md                    # Ten dokument
├── config.py                        # Ustawienia (ligi, parametry modeli)
├── data_loader.py                   # Pobieranie danych z internetu
├── database.py                      # Baza SQLite
├── features.py                      # Elo, forma, cechy XGBoost
├── model.py                         # Model XGBoost
├── dixon_coles.py                   # Model Dixon-Coles
├── predictor.py                     # Wykrywanie kolejki i prognozy
├── updater.py                       # Aktualizacja po kolejce
├── compare_models.py                # Porównanie modeli
├── prediction_history.py            # Archiwum prognoz (Streamlit)
│
├── data/
│   ├── matches.db                   # Baza meczów (SQLite)
│   ├── model.joblib                 # Wytrenowany XGBoost
│   ├── dixon_coles.joblib           # Wytrenowany Dixon-Coles
│   ├── metadata.json                # Metadane (daty aktualizacji)
│   └── raw/                         # Surowe pliki CSV (opcjonalnie)
│
└── output/
    ├── predictions.csv              # Aktualne prognozy
    ├── prediction_history.csv       # Archiwum prognoz (Streamlit)
    └── model_comparison.txt         # Raport porównania modeli
```

### Ważne pliki wyjściowe

**`output/predictions.csv`** — kolumny:

| Kolumna | Opis |
|---------|------|
| `model` | Użyty model (xgboost / dixon_coles) |
| `league` | Nazwa ligi |
| `date` | Data meczu |
| `home_team`, `away_team` | Drużyny |
| `predicted` | Prognoza (H / D / A) |
| `prediction_label` | Opis po polsku |
| `prob_H`, `prob_D`, `prob_A` | Prawdopodobieństwa (0–1) |
| `confidence` | Pewność = max(prob_H, prob_D, prob_A) |

---

## 10. Typowy cykl pracy

```
┌─────────────────────────────────────────────────────────┐
│  PIERWSZE URUCHOMIENIE (raz)                            │
│  python main.py init                                    │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  PROGNOZY NA KOLEJKĘ                                    │
│  python main.py predict                                 │
│  (lub: streamlit run app.py → Generuj prognozy)         │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  ROZGRANIE KOLEJKI (kilka dni)                          │
│  — czekasz na wyniki meczów —                           │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  AKTUALIZACJA                                           │
│  python main.py update                                  │
│  → dopisuje wyniki, trenuje modele, nowe prognozy       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           └──────► powtarzaj co tydzień
```

### Harmonogram (opcjonalnie)

Możesz zaplanować automatyczną aktualizację w **Harmonogramie zadań Windows**:

- Program: `python`
- Argumenty: `main.py update`
- Folder startowy: `C:\Data_Science\08_Machine_learning_1\football_predictor`
- Częstotliwość: np. każdy poniedziałek o 8:00

---

## 11. Jak czytać prognozy

### Przykład wiersza z `predictions.csv`

```
dixon_coles | Premier League | 2026-08-16
  Liverpool vs Bournemouth
  Prognoza: Wygrana gospodarzy (H:70% D:18% A:12%)
```

**Interpretacja:**
- Model przewiduje wygraną Liverpoolu (H).
- Pewność: 70% — model jest dość zdecydowany.
- 18% szans na remis, 12% na wygraną gości.

### Kiedy prognoza jest „mocna”, a kiedy „słaba"

| Pewność (confidence) | Interpretacja |
|----------------------|---------------|
| **> 60%** | Wyraźny faworyt — model widzi przewagę |
| **40–60%** | Umiarkowana pewność — mecz bardziej otwarty |
| **< 40%** | Niska pewność — wynik trudny do przewidzenia (remis lub mecz wyrównany) |

### Co program **nie** robi

- Nie przewiduje dokładnego wyniku (np. 2:1) — tylko H/D/A.
- Nie uwzględnia kontuzji, składów, kartek ani pogody.
- Nie jest poradą bukmacherską.

---

## 12. Rozwiązywanie problemów

### „Baza danych jest pusta"

```powershell
python main.py init
```

### „Brak nadchodzących meczów do prognozy"

- Sprawdź połączenie z internetem.
- Uruchom `python main.py refresh`.
- Możliwe, że między kolejkami nie ma jeszcze fixtures w football-data.co.uk.

### „Kolejka nie została jeszcze w pełni rozegrana"

Program czeka na wyniki wszystkich meczów z ostatniej prognozy.
Rozwiązania:
- Poczekaj, aż wszystkie mecze się rozegrają.
- Użyj `python main.py update --force` (w Streamlit: checkbox *Wymuś aktualizację*).

### Brak kursów / statystyk strzałów w bazie

```powershell
python main.py refresh
python main.py status
```

### Trening trwa bardzo długo

- XGBoost na pełnej bazie (~8000 meczów) może trwać 1–3 minuty — to normalne.
- Dixon-Coles trenuje się w ~1–2 sekundy.
- `compare` uruchamia oba modele — licz 2–4 minuty.

### Błąd `ModuleNotFoundError`

```powershell
pip install -r requirements.txt
```

### Streamlit nie otwiera przeglądarki

Wejdź ręcznie na adres wyświetlony w terminalu, zwykle:
`http://localhost:8501`

### Błąd pobierania CSV (timeout, 404)

- football-data.co.uk bywa chwilowo niedostępne — spróbuj ponownie za kilka minut.
- Sezon jeszcze nieopublikowany (np. `2627`) jest pomijany — to normalne.

---

## 13. Najczęstsze pytania

**Czy potrzebuję konta lub tokenu API?**
Nie. Domyślnie program korzysta z darmowych plików CSV na football-data.co.uk.

**Który model wybrać?**
Na start: `auto`. Jeśli chcesz decydować sam: uruchom `compare` i wybierz lepszy.

**Czy mogę dodać więcej lig?**
Tak — w pliku `config.py` w słowniku `LEAGUES` (wymaga kodów z football-data.co.uk).

**Czy prognozy zapisują się automatycznie?**
Tak — w `output/predictions.csv`. W Streamlit dodatkowo w `output/prediction_history.csv`.

**Czy mogę uruchomić program na innym komputerze?**
Tak — skopiuj folder `football_predictor` wraz z `data/` i `output/`, zainstaluj zależności i uruchom.

**Jak zresetować system od zera?**
Usuń folder `data/` (baza + modele) i uruchom `python main.py init`.

---

*Football Predictor — instrukcja obsługi. Ostatnia aktualizacja: sierpień 2026.*
