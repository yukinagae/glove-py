# glove-py

Practice: Python implementation of GloVe word embedding algorithm

## Dependencies

- Python3.6
- [Poetry](https://github.com/sdispater/poetry)

## Installation

```bash
cd glove-py
poetry install
```

### Download corpus data

```bash
$ cd data/
$ wget http://mattmahoney.net/dc/text8.zip
$ unzip text8.zip
$ rm text8.zip
$ ls
text8
```

If you want a smaller dataset to play with it, run the below commands.

```bash
$ cd data/
$ head -c 1279 text8 > small_text8 # First 1,279 bytes of the original dataset. I think this is small enough and it looks like the end of the sentence.
$ ls
small_text8
```

## Usage

```bash
cd glove-py
poetry run python main.py ./data/text8
```

## References

- [A GloVe implementation in Python](http://www.foldl.me/2014/glove-python/)
- [hans/glove.py](https://github.com/hans/glove.py)
