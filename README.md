# Predikce sekundární struktury proteinu pomocí strojového učení
Tato práce se zaměřuje na predikci sekundární struktury proteinů pomocí strojového učení, což je klíčová oblast bioinformatiky s významnými aplikacemi v lékařském výzkumu a farmaceutickém průmyslu. Sekundární struktura proteinů, zahrnující alfa-helixy, beta-listy a smyčky, je zásadní pro pochopení jejich funkcí a interakcí. Kombinace konvolučních neuronových sítí (CNN) a rekurentních neuronových sítí (RNN) byla navržena jako efektivní metoda pro zvýšení přesnosti predikce.

Práce dosáhla významného pokroku oproti tradičním metodám. Model kombinující CNN a RNN dosáhl přesnosti až 81 % při predikci osmi tříd sekundární struktury (Q8) na testovacím datasetu CB513, což představuje zlepšení o 7 % oproti čistým CNN. Tento přístup efektivně zachycuje prostorové i sekvenční vzorce v proteinových sekvencích, přičemž si zachovává nízkou výpočetní náročnost.

Součástí projektu byla také vývoj webové aplikace, která umožňuje uživatelům snadno využívat vyvinuté modely pro predikci sekundární struktury proteinů. Aplikace zahrnuje intuitivní uživatelské rozhraní, REST API pro integraci s dalšími bioinformatickými nástroji a vizualizaci výsledků. Výsledky byly validovány na standardních datasetech a ukazují robustnost modelů napříč různými typy proteinů.

# Running entire app
docker-compose up --build

web than can be accessed on port 3000 in browser

each model inside can be run using python 3.11
## Please rise an issue if bug or anything else seems to intervene with user experience 


