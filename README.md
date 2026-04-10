# Pipeline de Processamento de Imagens com OpenCV

Este projeto implementa um sistema de **processamento de imagens** utilizando OpenCV, com suporte a execução em imagens estáticas e em tempo real (webcam), além de visualização passo a passo dos resultados. Para organizar e reaproveitar o código, extraímos as funções principais nos scripts render_pipeline.py (com código de renderização das imagens) e image_transformes.py (que contém as funções de transformação)

---

# criar ambiente
python -m venv venv

# ativar (Linux/Windows)
source venv/bin/activate  # ou venv\Scripts\activate

# instalar dependências
pip install opencv-python numpy matplotlib notebook ipykernel

# executar
jupyter notebook

# executar o código
Abra o arquivo pipeline.ipynb e execute as células na ordem em que aparecem.