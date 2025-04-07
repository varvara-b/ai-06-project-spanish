<h3 align="center">Техническое описание модели</h3>

**TrOCR [(печатное распознавание)](http://microsoft/trocr-large-printed)**
Модель TrOCR, дообученная на [наборе данных SROIE](https://rrc.cvc.uab.es/?ch=13). Она была представлена в [статье](https://arxiv.org/abs/2109.10282) авторов Li et al. и впервые опубликована в [этом репозитории](https://github.com/microsoft/unilm/tree/master/trocr).

_Примечание: Команда, выпустившая TrOCR, не подготовила карту модели (model card), поэтому данное описание было составлено командой Hugging Face._  

<h5 align="center">**Описание модели**</h5>
TrOCR — это модель типа «кодировщик-декодировщик», состоящая из трансформера для обработки изображений (кодировщик) и трансформера для генерации текста (декодировщик). Визуальный кодировщик инициализирован на основе весов модели BEiT, а текстовый декодировщик — на основе RoBERTa.  

Изображения подаются в модель в виде последовательности фрагментов фиксированного размера (разрешение 16×16), которые линейно преобразуются в эмбеддинги. Перед передачей в слои трансформерного кодировщика добавляются позиционные эмбеддинги. Затем текстовый декодировщик авторегрессивно генерирует токены.

<h5 align="center">**Применение и ограничения**</h5>
Модель можно использовать для оптического распознавания символов (OCR) на изображениях с одной строкой текста. В [хабе моделей](https://huggingface.co/models?search=microsoft/trocr) можно найти другие дообученные версии для конкретных задач.

***
<h3 align="center">Документация хода работы</h3>
<div align="center">

[Журнал задач](https://docs.google.com/spreadsheets/d/1e6fI30tqwKYHyXy-QpfAwAXOJLQMZhI2/edit?usp=sharing&ouid=112407436546437674558&rtpof=true&sd=true)

</div>
