# Coverage Analysis for 100 Questions

## Current Dataset (28 documents)
- **XIII century (8 docs):** Chinggis Khan, Ögedei, Kublai, Yuan Dynasty, Mongol Empire
- **XVII-XIX century (6 docs):** Bogd Khan, Manchu period, traditional culture
- **XX century (10 docs):** Modern Mongolia, 1921 Revolution, 1990 Democracy, culture
- **Ancient period (2 docs):** General Mongolian history, Xiongnu (Хүннү)

## Question Distribution (100 questions)

### 1. Ancient Period (МЭӨ–900) - 20 questions
**Хүннү, Сяньби, Жужан (МЭӨ–600)** - 10 questions
- Xiongnu shanyu, Modu shanyu, burial culture, treaties, social structure, economy, military

**Түрэг, Уйгур, Киргиз (600–900)** - 10 questions  
- Göktürk inscriptions, Bilge Khan, Tonyukuk, Uyghur Khaganate, Manichaeism, Orkhon script

**Current Coverage:** ❌ VERY LIMITED (only 1 Xiongnu Wikipedia article)
**Gap:** Need 15-18 more ancient history sources

---

### 2. Medieval Period (900–1368) - 40 questions

**Хамаг Монгол – Тэмүжин (900–1206)** - 10 questions
- Khamag Mongol, Yesugei, Temujin's early life, tribal conflicts, unification

**Их Монгол улс, Чингис хаан (1206–1227)** - 10 questions
- Yassa law, decimal military system, Khwarezm campaign, postal system

**Монголын эзэнт гүрэн (1227–1294)** - 10 questions
- Ögedei, Möngke, Kublai, Karakorum, four khanates, Golden Horde, Ilkhanate

**Юань гүрэн (1271–1368)** - 10 questions
- Yuan administration, currency, Tibet relations, Green Turban Rebellion, Japan invasions

**Current Coverage:** ✅ GOOD for Chinggis/Yuan (8 docs)
**Gap:** Need more on pre-1206 tribal period and post-Yuan fragmentation

---

### 3. Late Medieval (1368–1757) - 30 questions

**Зүүн–Баруун Монгол, Даян хаан (1368–1636)** - 10 questions
- Dayan Khan, Six Tumens, Oirat-Khalkha conflicts, Altan Khan, Buddhism revival

**Ойрад–Зүүнгар улс (1636–1757)** - 10 questions
- Dzungar Khanate, Galdan Boshugtu, Qing-Dzungar wars, Oirat law code

**Манжийн үе (1691–1911)** - 10 questions
- Manchu rule, Khalkha submission, administrative structure, taxation, cultural policy

**Current Coverage:** ❌ VERY LIMITED (some Manchu period in textbook)
**Gap:** Need 25+ sources on this entire period

---

### 4. Modern Period (1911–1924) - 10 questions

**Үндэсний эрх чөлөөний хувьсгал (1911–1924)** - 10 questions
- 1911 independence, Bogd Khan government, 1919 Chinese occupation, 1921 Revolution, Baron Ungern

**Current Coverage:** ✅ EXCELLENT (Bogd Khan, 1921 Revolution, BNMAU founding)
**Gap:** Minimal - maybe add more on Baron Ungern and 1919-1921 period

---

## Priority Sources to Add

### HIGH PRIORITY (Ancient & Medieval)

1. **Хүннү улс (Xiongnu Empire)**
   - Wikipedia: Хүннү улс (expand current)
   - Модун шаньюй article
   - Хүннү-Хан гэрээ (treaties)

2. **Түрэг, Уйгур (Turkic & Uyghur)**
   - Көктүрк (Göktürk) Wikipedia
   - Билгэ хаан article
   - Уйгурын хаант улс
   - Орхон бичээс (Orkhon inscriptions)

3. **Тэмүжин/Early Chinggis**
   - Expand Chinggis Khan article (early life section)
   - Хамаг Монгол article
   - Есүхэй баатар

4. **Даян хаан period**
   - Даян хаан Wikipedia
   - Алтан хаан article
   - Зүүн-Баруун Монголын хуваагдал

5. **Зүүнгар (Dzungar)**
   - Зүүнгарын хаант улс Wikipedia
   - Галдан Бошигт article
   - Манж-Зүүнгарын дайн

6. **Манжийн үе (Manchu Period)**
   - Expand textbook Manchu sections
   - Халхын Манжид дагаар орсон түүх
   - XIX зууны Монгол

### MEDIUM PRIORITY

7. **Жужан, Сяньби**
   - Жужан улс Wikipedia
   - Сяньби article

8. **Киргиз**
   - Енисейн Киргиз article

9. **Four Khanates detail**
   - Алтан Ордын улс (Golden Horde)
   - Ил хаант улс (Ilkhanate)  
   - Цагаадайн улс (Chagatai)

### Sources to Scrape

**Wikipedia (mn.wikipedia.org):**
- Хүннү улс
- Модун шаньюй
- Көктүрк
- Билгэ хаан
- Уйгурын хаант улс
- Жужан
- Сяньби
- Даян хаан
- Алтан хаан
- Зүүнгарын хаант улс
- Галдан Бошигт
- Алтан Ордын улс
- Ил хаант улс

**Textbook sections to extract:**
- Chapter on Хүннү period (if exists)
- Chapter on Түрэг-Уйгур period
- Expanded Manchu period sections
- Даян хаан era sections

## Estimated Coverage After Adding Sources

| Period | Current | After Adding | Questions |
|--------|---------|--------------|-----------|
| Ancient (МЭӨ–900) | 5% | 70% | 20 |
| Medieval (900–1368) | 40% | 85% | 40 |
| Late Medieval (1368–1757) | 10% | 60% | 30 |
| Modern (1911–1924) | 90% | 95% | 10 |
| **TOTAL** | **30%** | **75%** | **100** |

## Action Plan

1. **Scrape Wikipedia articles** (13 articles) - 2-3 hours
2. **Extract textbook sections** - 1 hour  
3. **Clean and merge** - 1 hour
4. **Regenerate embeddings** - 30 minutes
5. **Test coverage** - 30 minutes

**Total effort:** ~5-6 hours to go from 30% to 75% coverage
