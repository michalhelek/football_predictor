# Football Predictor — instrukcja aplikacji webowej

Krótki przewodnik: **jak krok po kroku wygenerować prognozy** wyników meczów (H / D / A) dla 6 lig europejskich.

---

## Co zobaczysz w aplikacji

- **Panel boczny (lewa strona)** — tu uruchamiasz główne akcje.
- **Zakładki (góra strony)** — tu czytasz wyniki. Najważniejsza to **Prognozy**.

---

## Krok 1 — Pierwsze uruchomienie (tylko raz)

1. Otwórz aplikację w przeglądarce.
2. W **panelu bocznym** kliknij **🚀 Inicjalizacja bazy**.
3. Poczekaj **1–5 minut** (pasek postępu na górze strony).
4. Po zakończeniu zobaczysz komunikat sukcesu, np.:
   - liczba meczów w bazie (powinno być ok. **8000+**),
   - liczba meczów z kursami (powinno być ok. **8000+**).

**Co to robi:** program ładuje historię meczów z ostatnich sezonów — potrzebną do treningu modeli i prognoz.

**Sprawdzenie:** na dole panelu bocznego zobaczysz metryki **Mecze w bazie** i **Z kursami**. Jeśli obie wartości są większe od zera, możesz przejść do kroku 2.

> **Pierwsze uruchomienie na Streamlit Cloud:** przed kliknięciem *Inicjalizacja bazy* upewnij się, że w **Settings → Secrets** jest ustawiony token API (patrz sekcja *Problemy* poniżej).

---

## Krok 2 — Wybór modelu (opcjonalnie)

W panelu bocznym, nad przyciskami, wybierz **Model**:

| Opcja | Kiedy wybrać |
|-------|--------------|
| **Auto** | Domyślnie — zalecane na start |
| **Ensemble (DC + XGBoost)** | Zbalansowana prognoza (zalecane na co dzień) |
| **XGBoost** | Model uczenia maszynowego |
| **Dixon-Coles** | Model statystyczny |
| **Kursy bukmacherskie** | Tylko tam, gdzie są kursy w terminarzu (często mniej lig) |

Jeśli nie wiesz, co wybrać — zostaw **Auto** lub wybierz **Ensemble**.

---

## Krok 3 — Generowanie prognoz

1. W panelu bocznym kliknij **📊 Generuj prognozy**.
2. Poczekaj ok. **1 minuty**.
3. Po sukcesie zobaczysz komunikat, np. *„Zapisano 42 prognoz w 6 ligach”*.

**Co to robi:** program pobiera terminarz najbliższych meczów, trenuje modele na historii i zapisuje prognozy.

---

## Krok 4 — Odczytanie prognoz dla drużyn

1. Przejdź do zakładki **Prognozy** (pierwsza zakładka).
2. Na górze wybierz filtry:
   - **Liga** — np. *Premier League* albo *Wszystkie*,
   - **Od / Do** — zakres dat meczów.
3. Przewiń listę meczów. Dla każdego meczu zobaczysz:
   - **Gospodarze vs Goście**, datę, ligę, numer kolejki,
   - **Prognozę**: *Wygrana gospodarzy*, *Remis* lub *Wygrana gości*,
   - **Pewność** — jak bardzo model jest przekonany (pasek 0–100%),
   - **Wykres** — prawdopodobieństwa H / Remis / A.

### Co oznaczają skróty

| Symbol | Znaczenie |
|--------|-----------|
| **H** | Wygrana gospodarzy (Home) |
| **D** | Remis (Draw) |
| **A** | Wygrana gości (Away) |

### Pobranie prognoz do pliku

1. Otwórz zakładkę **Tabela prognoz**.
2. Kliknij **Pobierz CSV** — zapiszesz plik `predictions.csv` na dysk.

---

## Krok 5 — Co robić co tydzień (po rozegraniu kolejki)

Po weekendzie, gdy mecze z prognozowanej kolejki się zakończą:

1. W panelu bocznym kliknij **📥 Aktualizuj po kolejce**.
2. Poczekaj kilka minut — program dopisze wyniki do bazy i przeliczy modele.
3. Kliknij ponownie **📊 Generuj prognozy** — dostaniesz prognozy na **następną** kolejkę.

**Opcja zaawansowana:** zaznacz *Wymuś aktualizację*, jeśli część meczów została przełożona i chcesz zaktualizować mimo niepełnej kolejki.

---

## Przycisk „Odśwież dane” — kiedy używać

Przycisk **🔄 Odśwież dane** ponownie pobiera historię meczów z **football-data.co.uk** (pliki CSV) i uzupełnia bazę o **kursy bukmacherskie** oraz **statystyki** (strzały, rogi itd.).

**Na co dzień zwykle go nie potrzebujesz** — w cotygodniowej pracy wystarczą *Aktualizuj po kolejce* i *Generuj prognozy*. Przycisk *Aktualizuj po kolejce* i tak odświeża dane w tle.

### Kiedy kliknąć „Odśwież dane”

| Sytuacja | Dlaczego |
|----------|----------|
| W panelu **„Z kursami”** jest mało lub **0**, a mecze w bazie są | Brakuje kursów — ten przycisk dogrywa je z co.uk |
| Po *Inicjalizacji bazy* strona co.uk była chwilowo niedostępna | Ponowna próba pobrania CSV z kursami |
| Chcesz uzupełnić statystyki bez pełnej aktualizacji kolejki | Pobiera historię CSV, ale **nie trenuje modeli** i **nie generuje prognoz** |

### Różnica między przyciskami w panelu

| Przycisk | Co robi |
|----------|---------|
| **Inicjalizacja bazy** | Pierwsze ładowanie całej bazy (tylko raz na start) |
| **Generuj prognozy** | Prognozy na najbliższą kolejkę |
| **Aktualizuj po kolejce** | Wyniki + odświeżenie danych + retrening modeli |
| **Odśwież dane** | Tylko pobranie CSV i uzupełnienie kursów/statystyk |

> **Streamlit Cloud:** jeśli masz `FOOTBALL_DATA_SOURCE = "api"`, ten przycisk pobierze dane z API, które **nie ma kursów** — wtedy *Odśwież dane* nie pomoże. Kursy bierzesz z bazy **seed** przy inicjalizacji.

---

## Inne przydatne zakładki (opcjonalnie)

| Zakładka | Do czego służy |
|----------|----------------|
| **Terminarz** | Pełna lista meczów sezonu (rozegrane i zaplanowane) |
| **Analiza drużyn** | Forma, Elo i ostatnie mecze wybranej drużyny |
| **Historia prognoz** | Archiwum prognoz i trafność po rozegraniu meczów |
| **Porównanie modeli** | Który model jest dokładniejszy (przycisk *Porównaj modele* w panelu) |

---

## Typowy schemat pracy

```
Pierwszy raz:     Inicjalizacja bazy  →  Generuj prognozy  →  zakładka Prognozy
Co tydzień:       Aktualizuj po kolejce  →  Generuj prognozy  →  zakładka Prognozy
```

---

## Problemy — co zrobić

### „Brak prognoz” w zakładce Prognozy
Kliknij **Generuj prognozy** w panelu bocznym (krok 3).

### Błąd przy inicjalizacji (Streamlit Cloud)
1. Wejdź na https://www.football-data.org/client/register i skopiuj **API Token**.
2. W aplikacji: **Settings → Secrets** i wklej:
   ```toml
   FOOTBALL_DATA_ORG_TOKEN = "twój_token"
   FOOTBALL_DATA_SOURCE = "api"
   ```
3. Kliknij **Save**, poczekaj chwilę i ponów **Inicjalizacja bazy**.

### Mało meczów w bazie (np. 2000 zamiast 8000)
Sam token API nie zawiera kursów historycznych. Na Streamlit Cloud potrzebna jest baza **seed** w repozytorium (`data/matches_seed.db`). Po jej dodaniu ponów inicjalizację.

### Błąd przy generowaniu prognoz
- Sprawdź połączenie z internetem.
- Upewnij się, że baza nie jest pusta (krok 1).
- Spróbuj ponownie za minutę — API bywa chwilowo niedostępne.

---

*Football Predictor — instrukcja aplikacji webowej. Ostatnia aktualizacja: wrzesień 2026.*
