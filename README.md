`Github` не очень хорошо умеет в индексацию pdf, поэтому на сайте презентация выглядит криво в некоторых местах.

Чтобы все хорошо работало и выглядело, скопируйте себе командой 
```
git clone https://github.com/NonaryR/embeddings
```
Далее, распакуйте tar-архивы следующей командой
```
cd embeddings
find . -name '*.tar.gz' -exec tar -xzvf {} \;
```
Распакуются данные для обучения и файл с эмбеддингами

Код написан на `keras` с бэкэндом `tensorflow`, если вы предпочитаете `theano`, то результаты могут несколько отличаться
