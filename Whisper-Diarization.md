# Whisper-Diarization语音转文字➕声纹识别模型部署方案

## 安装

github项目地址：https://github.com/MahmoudAshraf97/whisper-diarization?tab=readme-ov-file

建议使用anaconda配置虚拟环境







`FFMPEG`并且`Cython`是安装要求的先决条件

```
pip install cython
```

或者

```
sudo apt update && sudo apt install cython3
```



```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg

# on Windows using WinGet (https://github.com/microsoft/winget-cli)
winget install ffmpeg
```



```
pip install -r requirements.txt
```



## 我的各项环境配置如下

```
Package                     Version
--------------------------- ------------
absl-py                     2.1.0
aiohappyeyeballs            2.3.5
aiohttp                     3.10.1
aiosignal                   1.3.1
alembic                     1.13.2
antlr4-python3-runtime      4.9.3
asteroid-filterbanks        0.4.0
asttokens                   2.4.1
async-timeout               4.0.3
attrs                       24.2.0
audioread                   3.0.1
av                          11.0.0
braceexpand                 0.1.7
certifi                     2024.7.4
cffi                        1.17.0
charset-normalizer          3.3.2
click                       8.1.7
cloudpickle                 3.0.0
coloredlogs                 15.0.1
colorlog                    6.8.2
comm                        0.2.2
contourpy                   1.2.1
ctc-forced-aligner          0.2
ctranslate2                 4.3.1
cycler                      0.12.1
Cython                      3.0.11
decorator                   5.1.1
deepmultilingualpunctuation 1.0.1
demucs                      4.1.0a3
Distance                    0.1.3
docker-pycreds              0.4.0
docopt                      0.6.2
dora_search                 0.1.12
editdistance                0.8.1
einops                      0.8.0
exceptiongroup              1.2.2
executing                   2.0.1
faster-whisper              1.0.0
filelock                    3.15.4
flatbuffers                 24.3.25
fonttools                   4.53.1
frozenlist                  1.4.1
fsspec                      2024.6.1
g2p-en                      2.1.0
gitdb                       4.0.11
GitPython                   3.1.43
grpcio                      1.65.4
huggingface-hub             0.19.3
humanfriendly               10.0
hydra-core                  1.2.0
HyperPyYAML                 1.2.2
idna                        3.7
inflect                     7.3.1
ipython                     8.26.0
ipywidgets                  8.1.3
jedi                        0.19.1
Jinja2                      3.1.4
jiwer                       3.0.4
joblib                      1.4.2
julius                      0.2.7
jupyterlab_widgets          3.0.11
kaldi-python-io             1.2.2
kaldiio                     2.18.0
kiwisolver                  1.4.5
lameenc                     1.7.0
lazy_loader                 0.4
Levenshtein                 0.25.1
librosa                     0.10.2.post1
lightning                   2.4.0
lightning-utilities         0.11.6
llvmlite                    0.43.0
loguru                      0.7.2
Mako                        1.3.5
Markdown                    3.6
markdown-it-py              3.0.0
MarkupSafe                  2.1.5
marshmallow                 3.21.3
matplotlib                  3.9.1.post1
matplotlib-inline           0.1.7
mdurl                       0.1.2
more-itertools              10.4.0
mpmath                      1.3.0
msgpack                     1.0.8
multidict                   6.0.5
nemo-toolkit                1.20.0
networkx                    3.3
nltk                        3.8.1
numba                       0.60.0
numpy                       1.23.5
omegaconf                   2.2.3
onnx                        1.16.2
onnxruntime                 1.18.1
openunmix                   1.3.0
optuna                      3.6.1
packaging                   24.1
pandas                      2.2.2
parso                       0.8.4
pexpect                     4.9.0
pillow                      10.4.0
pip                         24.0
plac                        1.4.3
platformdirs                4.2.2
pooch                       1.8.2
primePy                     1.3
prompt_toolkit              3.0.47
protobuf                    4.25.4
psutil                      6.0.0
ptyprocess                  0.7.0
pure_eval                   0.2.3
pyannote.audio              3.1.1
pyannote.core               5.0.0
pyannote.database           5.1.0
pyannote.metrics            3.2.1
pyannote.pipeline           3.0.1
pybind11                    2.13.1
pycparser                   2.22
pydantic                    1.10.17
pydub                       0.25.1
Pygments                    2.18.0
pyparsing                   3.1.2
python-dateutil             2.9.0.post0
pytorch-lightning           1.9.4
pytorch-metric-learning     2.6.0
pytz                        2024.1
PyYAML                      6.0.2
rapidfuzz                   3.9.6
regex                       2024.7.24
requests                    2.32.3
retrying                    1.3.4
rich                        13.7.1
ruamel.yaml                 0.18.6
ruamel.yaml.clib            0.2.8
sacremoses                  0.1.1
safetensors                 0.4.4
scikit-learn                1.5.1
scipy                       1.14.0
semver                      3.0.2
sentencepiece               0.2.0
sentry-sdk                  2.12.0
setproctitle                1.3.3
setuptools                  65.5.1
shellingham                 1.5.4
six                         1.16.0
smmap                       5.0.1
sortedcontainers            2.4.0
soundfile                   0.12.1
sox                         1.5.0
soxr                        0.4.0
speechbrain                 1.0.0
SQLAlchemy                  2.0.32
stack-data                  0.6.3
submitit                    1.5.1
sympy                       1.13.1
tabulate                    0.9.0
tensorboard                 2.17.0
tensorboard-data-server     0.7.2
tensorboardX                2.6.2.2
termcolor                   2.4.0
text-unidecode              1.3
texterrors                  0.5.1
threadpoolctl               3.5.0
tokenizers                  0.15.2
torch                       2.1.2
torch-audiomentations       0.11.1
torch-pitch-shift           1.2.4
torchaudio                  2.1.2
torchmetrics                1.4.1
tqdm                        4.66.5
traitlets                   5.14.3
transformers                4.39.3
treetable                   0.2.5
typeguard                   4.3.0
typer                       0.12.3
typing_extensions           4.12.2
tzdata                      2024.1
Unidecode                   1.3.8
urllib3                     2.2.2
wandb                       0.17.6
wcwidth                     0.2.13
webdataset                  0.1.62
Werkzeug                    3.0.3
wget                        3.2
wheel                       0.43.0
whisperx                    3.1.1
widgetsnbextension          4.0.11
wrapt                       1.16.0
yarl                        1.9.4
youtokentome                1.0.6
python版本：Python 3.10.14
```







## huggingface上需要下载的模型以及nemo需要下载的模型已经放在压缩包里



## 用法



```
python diarize.py -a AUDIO_FILE_NAME
```



如果您的系统有足够的 VRAM（>=10GB），您可以改用`diarize_parallel.py`，不同之处在于它与 Whisper 并行运行 NeMo，这在某些情况下是有益的，并且结果是相同的，因为这两个模型彼此不依赖。

## 命令行选项



- `-a AUDIO_FILE_NAME`：需要处理的音频文件的名称
- `--no-stem`：禁用源分离
- `--whisper-model`：ASR 使用的模型，默认为`medium.en`
- `--suppress_numerals`：以发音字母而非数字转录数字，提高对齐准确性
- `--device`：选择使用哪个设备，如果可用，默认为“cuda”
- `--language`：手动选择语言，在语言检测失败时有用
- `--batch-size`：批量推理的批量大小，如果内存不足则减少，非批量推理则设置为 0

