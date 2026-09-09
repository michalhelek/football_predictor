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

## Wykres Elo — co to jest i jak go interpretować

Wykres Elo znajdziesz w zakładce **Analiza drużyn** (po wyborze ligi i drużyny, po lewej stronie).

### Co to jest Elo?

**Elo** to liczbowa ocena **siły drużyny** obliczana na podstawie rozegranych meczów. Im wyższa wartość, tym drużyna jest uznawana za silniejszą.

- Każda drużyna startuje z poziomu ok. **1500 punktów** (średnia).
- Po **każdym meczu** rating się zmienia:
  - **wygrana** → Elo rośnie,
  - **porażka** → Elo spada,
  - **remis** → mała zmiana w górę lub w dół (zależy od siły rywala).
- Wynik z **silniejszym** rywalem daje większy wzrost przy wygranej; przegrana z dużo słabszym spada mocniej.
- Przy obliczeniach uwzględniana jest **przewaga własnego boiska** (gospodarze mają lekko łatwiej).

Wykres pokazuje, jak Elo drużyny **zmieniało się w czasie** — punkt na wykresie = rating **po** danym meczu.

### Jak czytać wykres?

| Co widzisz | Co to znaczy |
|------------|--------------|
| **Linia idzie w górę** | Dobra passa — wygrane lub remisy z mocnymi rywalami |
| **Linia idzie w dół** | Słabsze wyniki — porażki lub remisy z słabszymi |
| **Pozioma linia** | Stabilna forma — wyniki zgodne z oczekiwaniami |
| **Elo powyżej 1500** | Drużyna powyżej średniej w lidze |
| **Elo poniżej 1500** | Drużyna poniżej średniej w lidze |
| **Duży skok w górę/dół** | Niespodziany wynik względem siły rywala (np. wygrana z faworytem) |

Obok wykresu zobaczysz też liczbę **Aktualne Elo** — to ostatnia wartość z wykresu, używana m.in. przez model XGBoost przy prognozach.

### Na co uważać?

- Elo liczone jest **osobno w każdej lidze** — nie porównuj bezpośrednio drużyn z Premier League i La Ligi.
- To **nie jest tabela ligowa** — drużyna z wysokim Elo nie musi być liderem tabeli (np. ma mniej meczów z topowymi rywalami).
- Wykres opiera się na **historii w bazie** — im więcej sezonów w bazie, tym stabilniejszy obraz formy długoterminowej.
- Elo to **jedna z wielu cech** modelu — prognoza końcowa uwzględnia też formę, kursy i statystyki meczów.

---

## Zakładka Porównanie modeli — co to jest i jak ją interpretować

Zakładka **Porównanie modeli** pokazuje, **który sposób prognozowania był najtrafniejszy** na historycznych meczach z bazy. To pomaga zrozumieć różnice między modelami — nie musisz tego robić co tydzień.

### Jak uruchomić porównanie?

1. W **panelu bocznym** kliknij **⚖️ Porównaj modele**.
2. Poczekaj **1–3 minuty** (program trenuje modele na danych z bazy).
3. Przejdź do zakładki **Porównanie modeli** — zobaczysz wyniki.

Jeśli zakładka jest pusta, najpierw uruchom porównanie w panelu bocznym. Wynik zapisywany jest też w pliku `output/model_comparison.txt` i może się wyświetlić przy kolejnym wejściu w aplikację.

### Co zobaczysz na ekranie?

Na górze zakładki **5 wskaźników dokładności** (w procentach):

| Model | Co oznacza |
|-------|------------|
| **Dixon-Coles** | Model statystyczny oparty na bramkach i sile drużyn |
| **XGBoost** | Uczenie maszynowe (Elo, forma, kursy, statystyki) |
| **MLP** | Sieć neuronowa na tych samych cechach co XGBoost |
| **Kursy** | Benchmark — wybór faworyta bukmachera (najniższy kurs) |
| **Ensemble** | Średnia prawdopodobieństw z Dixon-Coles + XGBoost |

Pod spodem jest **raport tekstowy** z dokładnością, log-loss, liczbą ocenionych meczów i czasem treningu.

W linii **Lepszy model:** program wskazuje, który z powyższych miał **najwyższą dokładność** na zbiorze testowym.

### Jak to interpretować?

**Dokładność** — jaki procent meczów model trafił w wynik (H, D lub A).

| Dokładność | Co to znaczy |
|------------|--------------|
| **~33%** | Losowy typ (3 wyniki) — bardzo słabo |
| **~47–50%** | Typowy poziom modeli własnych (DC, XGBoost, MLP, Ensemble) |
| **~52–53%** | Kursy bukmacherskie — górny benchmark rynku |
| **Powyżej 55%** | Na dłuższej próbie rzadko się utrzymuje — piłka nożna jest niepewna |

**Log-loss** — im **niższy**, tym lepiej model ocenia prawdopodobieństwa (nie tylko trafia w wynik, ale też „pewność” prognozy).

**Lepszy model** — w trybie **Auto** przy *Generuj prognozy* aplikacja może korzystać z wyniku tego porównania, wybierając lepszy wariant.

### Jak powstaje porównanie?

Program dzieli mecze z bazy na dwa zbiory **chronologicznie** (nie losowo):

- **80% starszych meczów** → trening modeli,
- **20% najnowszych meczów** → test (sprawdzenie trafności).

Dzięki temu ocena jest realistyczna — modele nie „widzą przyszłości”.

### Na co uważać?

- Porównanie dotyczy **przeszłości** — lepszy model w teście nie gwarantuje trafności na **nadchodzących** meczach.
- **Kursy** działają tylko tam, gdzie w bazie są kursy bukmacherskie (historyczne mecze z co.uk).
- Uruchamianie porównania **obciąża API i czas** — wystarczy co kilka tygodni lub po dużej aktualizacji bazy.
- Do codziennych prognoz wystarczy **Ensemble** lub **Auto** — ta zakładka służy głównie do analizy i wyboru strategii.

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
